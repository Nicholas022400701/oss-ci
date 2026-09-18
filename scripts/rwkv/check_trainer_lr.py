# Verifies the fix for the undefined `lr` in RWKV-v7/train_temp/src/trainer.py.
# usage: python check_trainer_lr.py <upstream trainer.py> <fixed trainer.py>
# Works with the real pytorch_lightning if importable, otherwise installs tiny stubs
# for torch / pytorch_lightning so the callback module can be imported anywhere.
import sys, types, importlib.util, math

def ensure_importable():
    used = []
    try:
        import torch  # noqa
    except Exception:
        t = types.ModuleType('torch'); t.save = lambda *a, **k: None
        tu = types.ModuleType('torch.utils'); td = types.ModuleType('torch.utils.data'); td.DataLoader = object
        t.utils = tu; tu.data = td
        sys.modules['torch'] = t; sys.modules['torch.utils'] = tu; sys.modules['torch.utils.data'] = td
        used.append('torch')
    try:
        import pytorch_lightning  # noqa
    except Exception:
        pl = types.ModuleType('pytorch_lightning'); pl.Callback = type('Callback', (), {})
        plu = types.ModuleType('pytorch_lightning.utilities')
        plu.rank_zero_info = lambda *a, **k: None; plu.rank_zero_only = lambda f: f
        pl.utilities = plu
        sys.modules['pytorch_lightning'] = pl; sys.modules['pytorch_lightning.utilities'] = plu
        used.append('pytorch_lightning')
    print('stubbed modules:', used if used else 'none (real torch and pytorch_lightning imported)')

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def make_args(**over):
    a = types.SimpleNamespace(epoch_begin=0, epoch_steps=2520, warmup_steps=10, my_exit_tokens=0, ctx_len=512, real_bsz=16,
                              lr_final=6e-5, lr_init=6e-4, strategy='deepspeed_stage_2', proj_dir='/tmp/x', weight_decay=0.001,
                              wandb='', my_timestamp='t', run_name='r', magic_prime=0)
    for k, v in over.items():
        setattr(a, k, v)
    return a

class Trainer:
    def __init__(self, step):
        self.global_step = step
        self.is_global_zero = False
        self.optimizers = [types.SimpleNamespace(param_groups=[
            {'weight_decay': 0.0, 'my_lr_scale': 1.0, 'lr': 0.0},
            {'weight_decay': 0.0, 'my_lr_scale': 2.0, 'lr': 0.0},
            {'weight_decay': 0.001, 'my_lr_scale': 1.0, 'lr': 0.0},
        ])]

def step_lr(mod, step, **over):
    cb = mod.train_callback(make_args(**over))
    tr = Trainer(step)
    cb.on_train_batch_start(tr, None, None, 0)
    return tr.my_lr, tuple(g['lr'] for g in tr.optimizers[0].param_groups)

def main():
    ensure_importable()
    old = load(sys.argv[1], 'trainer_upstream')
    new = load(sys.argv[2], 'trainer_fixed')

    # 1. the default --my_exit_tokens 0 and any negative value crash upstream at the first batch
    for exit_tokens in (0, -1498226207, -1):
        try:
            step_lr(old, 0, my_exit_tokens=exit_tokens)
        except UnboundLocalError as e:
            print(f'upstream my_exit_tokens={exit_tokens}: {type(e).__name__}: {e}')
        else:
            raise SystemExit('expected upstream to fail, it did not')

    # 2. fixed file: my_exit_tokens=0 gives warmup then constant lr_init
    lr0, groups0 = step_lr(new, 0)
    assert math.isclose(lr0, 6e-4 * 0.01), lr0
    for step in (10, 11, 500, 2520, 100000):
        lr, groups = step_lr(new, step)
        assert lr == 6e-4, (step, lr)
        assert groups == (6e-4, 1.2e-3, 6e-4), groups
    print('fixed my_exit_tokens=0: warmup then constant lr_init, ok')

    # 3. fixed file: negative my_exit_tokens gives a finite half-strength cosine decay, ending at (lr_init + lr_final) / 2
    n = -1498226207
    lr_start, _ = step_lr(new, 10, my_exit_tokens=n)
    total_steps = abs(n) // (512 * 16)
    lr_end, _ = step_lr(new, total_steps + 10, my_exit_tokens=n, magic_prime=0)
    assert math.isclose(lr_start, 6e-4, rel_tol=1e-9), lr_start
    assert math.isclose(lr_end, (6e-4 + 6e-5) / 2, rel_tol=1e-6), lr_end
    print(f'fixed my_exit_tokens<0: lr {lr_start:.6e} -> {lr_end:.6e}, ok')

    # 4. positive my_exit_tokens: bit identical between upstream and fixed file
    checked = 0
    for exit_tokens in (1498226207, 100000000, 5000000):
        for lr_init, lr_final in ((6e-4, 6e-5), (1e-5, 1e-5), (3e-4, 0.0)):
            for warm in (-1, 0, 10, 50):
                for step in list(range(0, 60)) + [100, 1000, 12345, 200000, 10**9]:
                    a = step_lr(old, step, my_exit_tokens=exit_tokens, lr_init=lr_init, lr_final=lr_final, warmup_steps=warm)
                    b = step_lr(new, step, my_exit_tokens=exit_tokens, lr_init=lr_init, lr_final=lr_final, warmup_steps=warm)
                    assert a == b, (exit_tokens, lr_init, lr_final, warm, step, a, b)
                    checked += 1
    print(f'positive my_exit_tokens: {checked} (config, step) pairs bit identical between upstream and fixed file, ok')
    print('ALL_CHECKS_PASSED')

if __name__ == '__main__':
    main()
