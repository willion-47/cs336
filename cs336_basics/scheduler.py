import math

def get_lr_cosine_schedule(
    it: int, 
    max_learning_rate: float, 
    min_learning_rate: float, 
    warmup_iters: int, 
    cosine_cycle_iters: int
) -> float:
    """
    计算带预热的余弦退火学习率。
    
    it: 当前迭代次数 (t)
    max_learning_rate: 最大学习率 (alpha_max)
    min_learning_rate: 最小学习率 (alpha_min)
    warmup_iters: 预热步数 (T_w)
    cosine_cycle_iters: 总退火步数 (T_c)
    """

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)