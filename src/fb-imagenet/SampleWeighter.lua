require 'nn'

-----------
-- Weights backpropagation of different samples in batch by specified factors.
local SampleWeighter, SampleWeighter_parent = torch.class('nn.SampleWeighter', 'nn.Module')

function SampleWeighter:__init()
    SampleWeighter_parent.__init(self)
    self.factors = torch.Tensor() --set externally
end

function SampleWeighter:updateOutput(input)
   self.output = input
   return self.output
end

function SampleWeighter:updateGradInput(input, gradOutput)
    local factors = gradOutput:dim()==2 and self.factors:view(-1,1) or self.factors
    local L1orig = gradOutput:norm(1)
    gradOutput:cmul(factors:expandAs(gradOutput))
    local L1new = gradOutput:norm(1)
    gradOutput:mul(L1orig/L1new) --keep the L1-norm of gradient ("it's energy") for consistency
    self.gradInput = gradOutput
    return self.gradInput
end

function SampleWeighter:setFactors(factors)
    self.factors:resize(factors:size()):copy(factors)
end