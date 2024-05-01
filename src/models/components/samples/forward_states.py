import torch

class SingleForwardState:
    def __init__(self, sequences: torch.FloatTensor, padding_mask: torch.BoolTensor):
        self.sequences = sequences.clone()
        self.padding_mask = padding_mask.clone()
    
    @classmethod
    def init_with_zero_mask(cls, sequences: torch.FloatTensor):
        return cls(
            sequences, 
            sequences.data.new_zeros(sequences.size()[:-1], dtype=torch.bool)
        )
    
    @classmethod
    def clone_sfs(cls, input_single_forward_state): # sfs = SingleForwardState
        return cls(
            input_single_forward_state.sequences,
            input_single_forward_state.padding_mask
        )
    
    def __repr__(self):
        return f"SFS sequence:\n{repr(self.sequences)}\nSFS padding mask:\n{repr(self.padding_mask)}"
    

class TwoBranchForwardState:
    def __init__(self, main_seq: SingleForwardState, aggregates: SingleForwardState):
        self.main_seq = SingleForwardState.clone_sfs(main_seq)
        self.aggregates = SingleForwardState.clone_sfs(aggregates)

    @classmethod
    def clone_tbfs(cls, input_two_branch_forward_state): # tbfs = TwoBranchForwardState
        return cls(
            input_two_branch_forward_state.main_seq,
            input_two_branch_forward_state.aggregates
        )
    
    def __repr__(self):
        return f"2BFS main sequence:\n{repr(self.main_seq)}\n2BFS aggregates:\n{repr(self.padding_mask)}"