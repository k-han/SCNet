classdef ProposalLocalVote < dagnn.ElementWise
  properties
    opts = {}
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnlocalvote(inputs{1}, inputs{2}, inputs{3}, inputs{4}, inputs{5}, []) ;
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnlocalvote(inputs{1}, inputs{2}, inputs{3}, inputs{4}, inputs{5}, derOutputs{1}, 'inputSizes', obj.inputSizes) ;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derInputs{4} = [] ;
      derInputs{5} = [] ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = ProposalHoughVote(varargin)
      obj.load(varargin) ;
    end
  end
end
