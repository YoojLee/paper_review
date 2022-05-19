from torch.optim.lr_scheduler import ConstantLR, _LRScheduler, SequentialLR
from torch.optim.optimizer import Optimizer
import warnings

def decay_after_k_epoch(optim: Optimizer, n_epochs:int, init_lr: float, target_lr: float, decay_after:int=100):
    """
    Returns a sequential learning rate scheduler in which learning rate decay is executed after k epochs

    - Args
        optim (Optimizer): an optimizer for learnig rate decay to be applicated on
        n_epochs (int): number of training epochs
        init_lr (float): initial learning rate
        target_lr (float): target learning rate (After training, learning rate should be decayed to this target_lr)
        decay_after (int): an epoch after which the decay occur
    """
    scheduler1 = ConstantLR(optim, factor=1, total_iters=decay_after)
    scheduler2 = LinearDecayLR(optim, initial_lr = init_lr, target_lr=target_lr, total_iters=n_epochs-decay_after) # constant는 그냥 한 번 줄여주고 마는 거임.

    scheduler = SequentialLR(optim, schedulers=[scheduler1, scheduler2], milestones=[decay_after])
    return scheduler


class LinearDecayLR(_LRScheduler):
    """
    Custom LR Scheduler which linearly decay to a target learning rate.

    override 해줘야 하는 부분은 get_lr, _get_closed_form_lr
    """
    def __init__(self, optimizer:Optimizer, initial_lr: float, target_lr:float, total_iters:int, last_epoch:int=-1, verbose:bool=False):
        
        if initial_lr < 0:
            raise ValueError("Initial Learning rate expected to be a non-negative integer.")
            
        if target_lr < 0:
            raise ValueError("Target Learning rate expected to be a non-negative integer.")

        if target_lr > initial_lr:
            raise ValueError("Target Learning Rate must be larger than Initial Learning Rate.")

        self.init_lr = initial_lr
        self.target_lr = target_lr
        self.total_iters = total_iters
        self.substract_lr = self._get_decay_constant()
        
        super(LinearDecayLR, self).__init__(optimizer, last_epoch, verbose) # 부모클래스의 init에 필요한 arg 넘겨줌.
        # super init을 해도 self.base_lrs를 왜 상속을 못받는 거지?
        # 여기서 부모 메소드 init 과정에서 self.get_lr을 호출함. 근데 get_lr에는 self.substract_lr이 들어가 있고, 그건 다시 부모클래스 init이 "완료"되어야 하기 때문에 
        # 아래 명령을 실행한 적이 없음. 그래서 계속 그런 attribute이 없다고 뜨는 거임.
            
        
    def _get_decay_constant(self):
        return float((self.init_lr-self.target_lr)/self.total_iters)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the learning rate computed by the scheduler, "
                        "please use 'get_last_lr()'.", UserWarning)

        return [group['lr']-self.substract_lr for group in self.optimizer.param_groups]


class DelayedLinearDecayLR(LinearDecayLR):
    def __init__(self, optimizer:Optimizer, initial_lr: float, target_lr: float, total_iters:int, last_epoch:int=-1, decay_after:int=100, verbose:bool=False):
        self.decay_after = decay_after
        
        super(DelayedLinearDecayLR, self).__init__(optimizer, initial_lr, target_lr, total_iters, last_epoch, verbose)

    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the learning rate computed by the scheduler, "
                        "please use 'get_last_lr()'.", UserWarning)

        if self.decay_after <= self.last_epoch < (self.decay_after + self.total_iters): # 여기에 total iter도 고려해줘야함.
            return [group['lr']-self.substract_lr for group in self.optimizer.param_groups]

        else:
            return [group['lr'] for group in self.optimizer.param_groups]