using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random
using SymDoME


"""
    oneHotEncoding(feature::AbstractArray{<:Any, 1}, classes::AbstractArray{<:Any, 1})

Convert a categorical feature vector to one-hot encoded matrix representation.

# Arguments
- `feature`: Input feature vector with categorical values
- `classes`: Array of unique classes to encode

# Returns
- `Matrix{Bool}`: One-hot encoded matrix where each column represents a class
"""

function oneHotEncoding(feature:: AbstractArray{<:Any, 1}, classes:: AbstractArray{<:Any, 1})

    num_classes = length(classes)

    if num_classes <= 2
        oneHot = Matrix{Bool}(undef, length(feature), 1)
        oneHot[:, 1] = feature .== classes[1]
    else
        oneHot = Matrix{Bool}(undef, length(feature), num_classes)
        for i in 1:num_classes
            oneHot[:, i] = feature .== classes[i]
        end
    end

    return oneHot
end

"""
    oneHotEncoding(feature::AbstractArray{<:Any, 1})

Convert a categorical feature vector to one-hot encoded matrix using unique values as classes.
"""

function oneHotEncoding(feature:: AbstractArray{<:Any, 1}) 
    
    return oneHotEncoding(feature, unique(feature))
end

"""
    oneHotEncoding(feature::AbstractArray{Bool, 1})

Convert a boolean feature vector to matrix format for consistency.
"""

function oneHotEncoding(feature:: AbstractArray{Bool, 1}) 
    
    return reshape(feature, :, 1)
end

## Normalization Functions

"""
    calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})

Calculate minimum and maximum values for each feature to enable min-max normalization.

# Returns
- `Tuple`: (minimum_values, maximum_values) for each column
"""

function calculateMinMaxNormalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    mins = minimum(dataset, dims=1)
    maxs = maximum(dataset, dims=1)
    return (mins, maxs)
end

"""
    calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})

Calculate mean and standard deviation values for each feature to enable z-score normalization.

# Returns
- `Tuple`: (mean_values, std_values) for each column
"""

function calculateZeroMeanNormalizationParameters(dataset:: AbstractArray{<:Real, 2})
    
    meanValues = mean(dataset, dims=1)
    stdValues = std(dataset, dims=1)
    return (meanValues, stdValues)
end


"""
Normalize cyclic integer values into 2D coordinates using sine and cosine.

Args:
    values (AbstractVector{<:Real}): Vector of integer values in the range [0, range-1].
    range (Int): Maximum value (exclusive) defining the cyclic range.

Returns:
    Matrix of size (length(values), 2) where the first column is cos(angles)
    and the second column is sin(angles).
"""
function normalizeCyclic(values::AbstractVector{<:Real}, range::Int)
    # Validate input
    if !all(0 .<= values .<= range-1)
        error("Values must be in the range [0, $(range-1)]. Found: $(unique(values))")
    end
    if range < 1
        error("range must be positive. Received: $range")
    end

    # Compute angles (2π * values / range)
    angles = 2 * π * values / range

    # Create output matrix: [cos(angles), sin(angles)]
    return hcat(cos.(angles), sin.(angles))
end

"""
Normalize dataset using Min-Max scaling in place.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.
    normalizationParameters (NTuple{2, AbstractArray{<:Real,2}}): Tuple containing mins and maxs.

Returns:
    Normalized dataset (in place).
"""
function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    dataset .-= mins
    dataset ./= (maxs .- mins)
    dataset[:, vec(mins.==maxs)] .= 0
    return dataset

end

"""
Normalize dataset using Min-Max scaling in place, with parameters automatically calculated.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.

Returns:
    Normalized dataset (in place).
"""
function normalizeMinMax!(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end

"""
Normalize dataset using Min-Max scaling, returning a new copy.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.
    normalizationParameters (NTuple{2, AbstractArray{<:Real,2}}): Tuple containing mins and maxs.

Returns:
    New normalized dataset.
"""
function normalizeMinMax(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    mins, maxs = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- mins) ./ (maxs .- mins)
    normalized_dataset[:, vec(mins.==maxs)] .= 0;
    return normalized_dataset

end

"""
Normalize dataset using Min-Max scaling, returning a new copy.
Parameters are automatically calculated.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.

Returns:
    New normalized dataset.
"""
function normalizeMinMax(dataset:: AbstractArray{<: Real, 2})
    
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeMinMax(normalized_dataset, normalizationParameters)
end

"""
Normalize dataset using Zero-Mean scaling in place.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.
    normalizationParameters (NTuple{2, AbstractArray{<:Real,2}}): Tuple containing mean values and std deviations.

Returns:
    Normalized dataset (in place).
"""
function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanv, stdv = normalizationParameters
    dataset .-= meanv
    dataset ./= stdv
    dataset[:, vec(stdv.==0)] .= 0;
    return dataset
end

"""
Normalize dataset using Zero-Mean scaling in place.
Parameters are automatically calculated.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.

Returns:
    Normalized dataset (in place).
"""
function normalizeZeroMean!(dataset:: AbstractArray{<: Real, 2})
    
    normvalues = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normvalues)
end

"""
Normalize dataset using Zero-Mean scaling, returning a new copy.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.
    normalizationParameters (NTuple{2, AbstractArray{<:Real,2}}): Tuple containing mean values and std deviations.

Returns:
    New normalized dataset.
"""
function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2}, normalizationParameters:: NTuple{2, AbstractArray{<: Real, 2}})
    
    meanValues, stdValues = normalizationParameters
    normalized_dataset = copy(dataset)
    normalized_dataset .= (normalized_dataset .- meanValues) ./ stdValues
    normalized_dataset[:, vec(stdValues.==0)] .= 0;
    return normalized_dataset
end

"""
Normalize dataset using Zero-Mean scaling, returning a new copy.
Parameters are automatically calculated.

Args:
    dataset (AbstractArray{<:Real,2}): Data matrix to normalize.

Returns:
    New normalized dataset.
"""
function normalizeZeroMean(dataset:: AbstractArray{<: Real, 2})
    
    norm_values = calculateZeroMeanNormalizationParameters(dataset)
    normalized_dataset = copy(dataset)
    return normalizeZeroMean(normalized_dataset, norm_values)
end

# Functions to classify outputs: probabilities -> bools

"""
Classify outputs from probabilities to boolean values using a threshold.

Args:
    outputs (AbstractArray{<:Real,1}): Vector of probabilities.
    threshold (Real, optional): Classification threshold. Default = 0.5.

Returns:
    Boolean vector indicating predicted classes.
"""
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end

"""
Classify outputs from probability matrices to boolean values.

Args:
    outputs (AbstractArray{<:Real,2}): Matrix of probabilities per instance.
    threshold (Real, optional): Threshold for binary classification. Default = 0.5.

Returns:
    Boolean matrix with predicted classes.
"""
function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 

    dims = size(outputs)
    
    if dims[2] == 1
        return reshape(classifyOutputs(outputs[:]; threshold), : ,1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2) # Get index of the class with the highest probability
        outputs = falses(dims)
        outputs[indicesMaxEachInstance] .= true
        return outputs  
    end
end

# Functions to compute the accuracy of model outputs.

# Case 1: target vector (bool) and output vector (bool)

"""
Compute accuracy given boolean outputs and boolean targets (1D).

Args:
    outputs (AbstractArray{Bool,1}): Predicted classes.
    targets (AbstractArray{Bool,1}): True classes.

Returns:
    Mean accuracy as a Float64.
"""
function accuracy(outputs:: AbstractArray{Bool, 1}, targets:: AbstractArray{Bool, 1})
    
    return mean(targets .== outputs)
end

# Case 2: target matrix (bool) and output matrix (bool)

"""
Compute accuracy given boolean outputs and targets (2D).

Args:
    outputs (AbstractArray{Bool,2}): Predicted classes matrix.
    targets (AbstractArray{Bool,2}): True classes matrix.

Returns:
    Mean accuracy as a Float64.
"""
function accuracy(outputs:: AbstractArray{Bool, 2}, targets:: AbstractArray{Bool, 2})

    dims = size(targets)

    if dims[2] == 1
        return accuracy(outputs[:], targets[:])

    elseif dims[2] > 2

        classComparison = targets .== outputs # If class predicted correctly, all row elements are true (true==true, false==false, etc.)
        correctClassifications = all(classComparison, dims=2) 
        return mean(correctClassifications)

    end
end

# Case 3: target vector (bool) and probability outputs

"""
Compute accuracy given probability outputs and boolean targets (1D).

Args:
    outputs (AbstractArray{<:Real,1}): Predicted probabilities.
    targets (AbstractArray{Bool,1}): True classes.
    threshold (Real, optional): Classification threshold. Default = 0.5.

Returns:
    Mean accuracy as a Float64.
"""
function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    
    return accuracy(outputs .>= threshold, targets)
end


# Fourth case: target matrix (bool) and probability outputs from the ANN.

"""
Compute accuracy given probability outputs (2D) and boolean targets (2D).

Args:
    outputs (AbstractArray{<:Real,2}): Predicted probability matrix from ANN.
    targets (AbstractArray{Bool,2}): True classes matrix.
    threshold (Real, optional): Classification threshold. Default = 0.5.

Returns:
    Mean accuracy as a Float64.
"""
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5) 
           
    dims = size(targets)

    if dims[2] == 1
        return accuracy(outputs[:], targets[:]; threshold=threshold)

    elseif dims[2] > 2
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
        
    end
end


"""
    buildClassANN(numInputs::Int, topology::AbstractArray{<:Int, 1}, numOutputs::Int; transferFunctions)

Build a neural network for classification tasks using Flux.jl.

# Arguments
- `numInputs`: Number of input features
- `topology`: Array defining hidden layer sizes
- `numOutputs`: Number of output classes
- `transferFunctions`: Activation functions for each layer (default: σ)

# Returns
- `Chain`: Flux neural network model ready for training
"""

function buildClassANN(numInputs:: Int, topology:: AbstractArray{<:Int, 1}, numOutputs:: Int;
     transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    
    ann = Chain()
    numInputsLayer = numInputs
    
    # Construir capas ocultas usando las funciones de transferencia correspondientes
    for (numOutputsLayer, transferFunction) in zip(topology, transferFunctions)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction))
        numInputsLayer = numOutputsLayer
    end

    if numOutputs == 1 # problema de clasificacion binaria
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ));
    else # problema de clasificación multiclase
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs), softmax);
    end

    return ann
end

"""
    holdOut(N::Int, P::Real)

Split dataset indices into training and test sets using holdout method.

# Arguments
- `N`: Total number of samples
- `P`: Proportion for test set (0.0 to 1.0)

# Returns
- `Tuple`: (training_indices, test_indices)
"""

function holdOut(N::Int, P::Real)
    @assert 0 <= P <= 1 "P must be in range [0, 1]"
    rng = MersenneTwister(42)  # set seed
    indices = randperm(rng, N)  # use generator with seed
    split_point = floor(Int, N * (1 - P))
    return (indices[1:split_point], indices[split_point+1:end])
end

"""
Split dataset indices into training, validation, and test sets using hold-out strategy.

Args:
    N (Int): Total number of samples.
    Pval (Real): Proportion of samples for validation set (0 ≤ Pval ≤ 1).
    Ptest (Real): Proportion of samples for test set (0 ≤ Ptest ≤ 1).

Returns:
    Tuple of (train_indices, val_indices, test_indices).
"""
function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert 0 <= Pval + Ptest <= 1 "Pval + PTest must be in range [0, 1]"
    (train_test, test) = holdOut(N, Ptest)  # Separate test dataset
    Pval_adjusted = Pval / (1 - Ptest)  # Adjust Pval for the remaining dataset
    train, val = holdOut(length(train_test), Pval_adjusted)  # Separate validation dataset from training dataset
    return (train_test[train], train_test[val], test)  # Return the three subsets
end


"""
    trainClassANN(topology, trainingDataset; validationDataset, testDataset, maxEpochs, learningRate)

Train a classification neural network with optional validation and early stopping.

# Arguments
- `topology`: Hidden layer architecture
- `trainingDataset`: Tuple of (inputs, targets) for training
- `validationDataset`: Optional validation data for early stopping
- `testDataset`: Optional test data for evaluation
- `maxEpochs`: Maximum training epochs (default: 1000)
- `learningRate`: Learning rate for Adam optimizer (default: 0.01)

# Returns
- `Tuple`: (trained_model, training_losses, validation_losses, test_losses)
"""

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0,size(trainingDataset[2],2))),
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    trainingInputs = Float32.(trainingDataset[1]') # Convert inputs to float32
    validationInputs = Float32.(validationDataset[1]') # Convert inputs to float32
    testInputs = Float32.(testDataset[1]') # Convert inputs to float32

    trainingTargets = trainingDataset[2]' # Extract labels
    validationTargets = validationDataset[2]' # Extract labels
    testTargets = testDataset[2]' # Extract labels
    
    numTrainingInputs = size(trainingInputs, 1)
    numTrainingOutputs = size(trainingTargets, 1)
    
    ann = buildClassANN(numTrainingInputs, topology, numTrainingOutputs; transferFunctions) # build ANN
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y) # loss function
    
    opt_state = Flux.setup(Adam(learningRate), ann) 

    trainingLosses = Float32[] # losses array
    validationLosses = Float32[] # validation losses
    testLosses = Float32[] # test losses array

    push!(trainingLosses, loss(ann, trainingInputs, trainingTargets)) # append epoch 0 loss 

    if !isempty(validationInputs)
        push!(validationLosses, loss(ann, validationInputs, validationTargets))
        bestValidationLoss = validationLosses[1]
        bestAnn = deepcopy(ann)
        epochsWithoutImprovement = 0
    end
    if !isempty(testInputs)
        push!(testLosses, loss(ann, testInputs, testTargets))
    end
    
    for epoch in 1:maxEpochs
      
        Flux.train!(loss, ann, [(trainingInputs, trainingTargets)], opt_state)
        currentTrainingLoss = loss(ann, trainingInputs, trainingTargets)
        push!(trainingLosses, currentTrainingLoss)
        
        if currentTrainingLoss <= minLoss
            break
        end

        # Case: test set
        if !isempty(testInputs)
            currentTestLoss = loss(ann, testInputs, testTargets)
            push!(testLosses, currentTestLoss)
        end

    
        # Validation
        if !isempty(validationInputs)
            currentValidationLoss = loss(ann, validationInputs, validationTargets)
            push!(validationLosses, currentValidationLoss)
            
            # early stopping
            if currentValidationLoss < bestValidationLoss
       
                bestValidationLoss = currentValidationLoss
                bestAnn = deepcopy(ann)
                epochsWithoutImprovement = 0  # Restart epochs with no improvement counter
            else
                epochsWithoutImprovement += 1
            end

            if epochsWithoutImprovement >= maxEpochsVal
                println("Early stop in epoch $epoch ")
                break
            end
        end
    end
    
    if !isempty(validationInputs)
        return (bestAnn, trainingLosses, validationLosses, testLosses)
    else
        return (ann, trainingLosses, validationLosses, testLosses)
    end
end;
   

"""
Train an Artificial Neural Network (ANN) classifier with optional validation and test datasets.

This function reshapes target vectors into column format and delegates training to the main
`trainClassANN` implementation. Supports early stopping with validation.

# Arguments
- `topology`: Hidden layer architecture as an array of integers.
- `trainingDataset`: Tuple of (inputs, boolean_targets).
- `validationDataset`: Tuple of (inputs, boolean_targets) for validation. Defaults to empty dataset.
- `testDataset`: Tuple of (inputs, boolean_targets) for testing. Defaults to empty dataset.
- `transferFunctions`: Activation functions for each layer (default: σ for all layers).
- `maxEpochs`: Maximum number of training epochs (default: 1000).
- `minLoss`: Minimum loss threshold for early stopping (default: 0.0).
- `learningRate`: Learning rate for optimizer (default: 0.01).
- `maxEpochsVal`: Maximum number of epochs without validation improvement before early stopping (default: 20).

# Returns
- Trained ANN classifier model (from the underlying `trainClassANN` implementation).
"""
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0)),
    testDataset:: Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
   (Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)),
   falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 

    # Reshape targets into column vectors
    newTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    newValidationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    newTestDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    return trainClassANN(topology, newTrainingDataset; validationDataset=newValidationDataset, testDataset=newTestDataset, 
                         transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, 
                         learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end


"""
    confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

Calculate comprehensive classification metrics from binary predictions.

# Arguments
- `outputs`: Predicted boolean classifications
- `targets`: True boolean labels

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

    true_positives = sum(outputs .& targets)
    false_positives = sum(outputs .& .!targets)
    true_negatives = sum(.!outputs .& .!targets)
    false_negatives = sum(.!outputs .& targets)

    accuracy = (true_positives + true_negatives) / length(outputs) # precision
    fail_rate = (false_positives + false_negatives) / length(outputs) # tasa de fallo
    recall = (true_positives == 0 && false_negatives == 0) ? 1 : true_positives / (true_positives + false_negatives) # sensibilidad
    especificity = (true_negatives == 0 && false_positives == 0) ? 1 : true_negatives / (true_negatives + false_positives) # especificidad
    precision = (true_positives == 0 && false_positives == 0) ? 1 : true_positives / (true_positives + false_positives) # valor predictivo positivo
    npv = (true_negatives == 0 && false_negatives == 0) ? 1 : true_negatives / (true_negatives + false_negatives) # valor predictivo negativo
    f1 = (recall == 0 && precision == 0) ? 0 : 2 * (precision * recall) / (precision + recall) # f1 score
    confussion_matrix = [true_negatives false_positives; false_negatives true_positives] # matriz de confusión

    return (accuracy, fail_rate, recall, especificity, precision, npv, f1, confussion_matrix)
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

Calculate confusion matrix metrics from real-valued outputs using a classification threshold.

# Arguments
- `outputs`: Model output probabilities or scores
- `targets`: True boolean labels
- `threshold`: Classification threshold (default: 0.5)

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""
function confusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)

    outputs = outputs .> threshold
    return confusionMatrix(outputs, targets)

end

"""
    confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Calculate multiclass confusion matrix metrics with optional class weighting.

# Arguments
- `outputs`: Predicted boolean classifications (one-hot encoded)
- `targets`: True boolean labels (one-hot encoded)
- `weighted`: Use class-weighted averaging if true, uniform averaging if false

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""
function confusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)

    num_classes = size(targets, 2)

    if size(outputs, 2) == size(targets, 2) && size(outputs, 2) > 2

        recall = zeros(num_classes) # Reserve memory for sensitivity
        especificity = zeros(num_classes) # Reserve memory for specificity
        precision = zeros(num_classes) # Reserve memory for positive predictive value
        npv = zeros(num_classes) # Reserve memory for negative predictive value
        f1 = zeros(num_classes) # Reserve memory for f1 score

        for i in 1:size(outputs, 2)
            _, _, recall[i], especificity[i], precision[i], npv[i], f1[i], _ = confusionMatrix(outputs[:, i], targets[:, i])
        end

        confussion_matrix = [sum(outputs[:, j] .& targets[:, i]) for i in 1:num_classes, j in 1:num_classes]

        if weighted
            w = vec(sum(targets, dims=1))/size(targets, 1) # Weight for each class
        else
            w = repeat([1/num_classes], num_classes) # Uniform weight (= arithmetic mean)
        end

        # Calculate metrics according to class weight (weighted or arithmetic mean) 
        recall = sum(recall .* w)
        especificity = sum(especificity .* w)
        precision = sum(precision .* w)
        npv = sum(npv .* w)
        f1 = sum(f1 .* w)
        acc = accuracy(outputs, targets)
        fail_rate = 1 - acc

        return(acc, fail_rate, recall, especificity, precision, npv, f1, confussion_matrix)

    elseif size(outputs, 2) == size(targets, 2) && size(outputs, 2) == 1 # Binary classification
        return confusionMatrix(outputs[:], targets[:])
    end
end

"""
    confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

Calculate multiclass confusion matrix metrics from real-valued outputs with threshold classification.

# Arguments
- `outputs`: Model output probabilities or scores (multiclass)
- `targets`: True boolean labels (one-hot encoded)
- `threshold`: Classification threshold (default: 0.5)
- `weighted`: Use class-weighted averaging if true, uniform averaging if false

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""
function confusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)

    outputs = classifyOutputs(outputs; threshold=threshold)
    return confusionMatrix(outputs, targets, weighted=weighted)

end

"""
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)

Calculate confusion matrix metrics from categorical labels with specified classes.

# Arguments
- `outputs`: Predicted categorical labels
- `targets`: True categorical labels
- `classes`: Array of possible class labels
- `weighted`: Use class-weighted averaging if true, uniform averaging if false

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1}; weighted::Bool=true)

    @assert(all([in(label, classes) for label in vcat(targets, outputs)])) # Check that labels are correct
    @assert size(outputs, 1) == size(targets, 1)

    outputs = oneHotEncoding(outputs, classes)
    targets = oneHotEncoding(targets, classes)

    return confusionMatrix(outputs, targets, weighted=weighted)
end

"""
    confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)

Calculate confusion matrix metrics from categorical labels with automatically detected classes.

# Arguments
- `outputs`: Predicted categorical labels
- `targets`: True categorical labels
- `weighted`: Use class-weighted averaging if true, uniform averaging if false

# Returns
- `Tuple`: (accuracy, error_rate, recall, specificity, precision, npv, f1_score, confusion_matrix)
"""
function confusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)

    classes = unique(vcat(outputs, targets))
    return confusionMatrix(outputs, targets, classes, weighted=weighted)
end

"""
    printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Print formatted confusion matrix results for multiclass boolean predictions.

# Arguments
- `outputs`: Predicted boolean classifications (one-hot encoded)
- `targets`: True boolean labels (one-hot encoded)
- `weighted`: Use class-weighted averaging if true, uniform averaging if false
"""
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Last element is the confusion matrix
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

Print formatted confusion matrix results for multiclass real-valued predictions.

# Arguments
- `outputs`: Model output probabilities or scores (multiclass)
- `targets`: True boolean labels (one-hot encoded)
- `weighted`: Use class-weighted averaging if true, uniform averaging if false
"""
function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Last element is the confusion matrix
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)

Print formatted confusion matrix results for categorical predictions with specified classes.

# Arguments
- `outputs`: Predicted categorical labels
- `targets`: True categorical labels
- `classes`: Array of possible class labels
- `weighted`: Use class-weighted averaging if true, uniform averaging if false
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    cm = confusionMatrix(outputs, targets, classes; weighted=weighted)
    println("Confusion Matrix:\n", cm[end])  # Last element is the confusion matrix
    println("Accuracy: ", cm[1])
    println("Error Rate: ", cm[2])
    println("Sensitivity: ", cm[3])
    println("Specificity: ", cm[4])
    println("Positive Predictive Value (PPV): ", cm[5])
    println("Negative Predictive Value (NPV): ", cm[6])
    println("F1 Score: ", cm[7])
end

"""
    printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)

Print formatted confusion matrix results for categorical predictions with automatically detected classes.

# Arguments
- `outputs`: Predicted categorical labels
- `targets`: True categorical labels
- `weighted`: Use class-weighted averaging if true, uniform averaging if false
"""
function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))  # Extract classes automatically
    printConfusionMatrix(outputs, targets, classes; weighted=weighted)
end



"""
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

Train a DoME (Decision tree with Optimization and Multiple Evaluations) classifier for binary classification.

# Arguments
- `trainingDataset`: Tuple containing training inputs and boolean target vector
- `testInputs`: Test input data matrix
- `maximumNodes`: Maximum number of nodes allowed in the decision tree

# Returns
- `Array{Float64}`: Predicted outputs for test instances
"""
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
                        testInputs::AbstractArray{<:Real,2},
                        maximumNodes::Int)
    
    (trainingInputs, trainingTargets) = trainingDataset

    trainingInputs = Float64.(trainingInputs)
    testInputs = Float64.(testInputs)

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes) 
    testOutputs = evaluateTree(model, testInputs)

    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end

    return testOutputs  
end

"""
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

Train a DoME classifier for multiclass classification using one-vs-all strategy.

# Arguments
- `trainingDataset`: Tuple containing training inputs and boolean target matrix (one-hot encoded)
- `testInputs`: Test input data matrix
- `maximumNodes`: Maximum number of nodes allowed in each decision tree

# Returns
- `Array{Float64}`: Predicted output matrix for test instances (multiclass probabilities)
"""
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
                        testInputs::AbstractArray{<:Real,2},
                        maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset

    if size(trainingTargets, 2) == 1
        # If there's only one column, treat it as binary classification
        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes)
        return reshape(testOutputs, :, 1)

    elseif size(trainingTargets, 2) == 2
        # For two columns, use one of them (e.g., the first one)
        testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets[:, 1])), testInputs, maximumNodes)
        return reshape(testOutputs, :, 1)
    
    else 
        # "One-vs-all" strategy for more than two columns
        num_instances = size(testInputs, 1)
        num_classes = size(trainingTargets, 2)
        testOutputsMatrix = Array{Float64}(undef, num_instances, num_classes)

        for i in 1:num_classes
            testOutputs = trainClassDoME((trainingInputs, vec(trainingTargets[:, i])), testInputs, maximumNodes)
            testOutputsMatrix[:, i] = testOutputs
        end 
        return testOutputsMatrix
    end
end

"""
    trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)

Train a DoME classifier for categorical classification with automatic class detection.

# Arguments
- `trainingDataset`: Tuple containing training inputs and categorical target vector
- `testInputs`: Test input data matrix
- `maximumNodes`: Maximum number of nodes allowed in the decision tree

# Returns
- `Array`: Predicted categorical labels for test instances
"""
function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    testInputs::AbstractArray{<:Real,2},
    maximumNodes::Int)

    (trainingInputs, trainingTargets) = trainingDataset
    classes = unique(trainingTargets)

    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1))

    testOutputsDoME = trainClassDoME(
    (trainingInputs, oneHotEncoding(trainingTargets, classes)),
    testInputs, maximumNodes)

    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)

    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]

        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    elseif length(classes) > 2
        for numClass in axes(classes, 1)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end

    return testOutputs
end


"""
    crossvalidation(N::Int64, k::Int64)

Generate k-fold cross-validation indices for N samples.

# Arguments
- `N`: Total number of samples
- `k`: Number of folds

# Returns
- `Array{Int64}`: Array of fold assignments (1 to k) for each sample
"""
function crossvalidation(N::Int64, k::Int64)

    folds = 1:k # number of folds
    k_folds = repeat(folds, Int(ceil(N/k))) # number of elements in each fold
    k_folds = k_folds[1:N] # remove the extra elements
    n_folds = shuffle!(k_folds) # shuffle the elements in each fold
    return n_folds

end

"""
    crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)

Generate stratified k-fold cross-validation indices for binary classification.

Ensures balanced representation of both classes across folds. Returns `nothing` if any class has fewer than 10 samples.

# Arguments
- `targets`: Boolean target vector for binary classification
- `k`: Number of folds

# Returns
- `Array{Int64}` or `nothing`: Stratified fold assignments, or nothing if insufficient samples per class
"""
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)

    # Ensure each class has at least 10 representatives
    min_class_count = min(sum(targets), sum(.!targets))
    if min_class_count < 10     
       return
    end
     
    indices = collect(1:length(targets))
    indices[targets] = crossvalidation(sum(targets), k) # assign each row a value from the folds list
    indices[.!targets] = crossvalidation(sum(.!targets), k) # Call the previous function with the number of negative instances
    return indices
    
 end

"""
    crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

Generate stratified k-fold cross-validation indices for multiclass classification.

Ensures balanced representation of all classes across folds. Returns `nothing` if any class has fewer than 10 samples.

# Arguments
- `targets`: Boolean target matrix (one-hot encoded) for multiclass classification
- `k`: Number of folds

# Returns
- `Array{Int64}` or `nothing`: Stratified fold assignments, or nothing if insufficient samples per class
"""
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)

    # Ensure each class has at least 10 representatives
    class_counts = vec(sum(targets, dims=1))  # Number of instances per class

    min_class_count = minimum(class_counts)  # Minimum instances in any class
    if min_class_count < 10
        return
    end

    indices = collect(1:size(targets, 1))
    for i in axes(targets, 2)
        indices[targets[:,i]] = crossvalidation(sum(targets[:,i]), k) # assign each row a value from the folds list
    end

    return indices
end

"""
    crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)

Generate stratified k-fold cross-validation indices for categorical targets.

Automatically converts categorical targets to one-hot encoding before applying stratified cross-validation.

# Arguments
- `targets`: Categorical target vector
- `k`: Number of folds

# Returns
- `Array{Int64}` or `nothing`: Stratified fold assignments, or nothing if insufficient samples per class
"""
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64) 

    return crossvalidation(oneHotEncoding(targets), k)

end


"""
    normalizeFold(train_inputs, val_inputs, test_inputs; cont_cols, key_col, cat_cols, bool_col, cat_classes)

Normalize train, validation, and test datasets with mixed data types using different normalization strategies.

Applies z-score normalization to continuous features, cyclic normalization to temporal features, 
one-hot encoding to categorical features, and preserves boolean features. Normalization parameters 
are computed from training data and applied consistently across all sets.

# Arguments
- `train_inputs`: Training input matrix
- `val_inputs`: Validation input matrix
- `test_inputs`: Test input matrix
- `cont_cols`: Column indices for continuous variables (default: [1,2,3,4,5,7,8,10,11,13,14])
- `key_col`: Column index for cyclic/temporal variable (default: 6)
- `cat_cols`: Column index for categorical variable (default: 12)
- `bool_col`: Column index for boolean variable (default: 9)
- `cat_classes`: Array of categorical class labels (default: [4,3,1,5,0])

# Returns
- `Tuple`: (normalized_train, normalized_validation, normalized_test) as Float32 matrices
"""
function normalizeFold(train_inputs::AbstractArray{<:Real,2}, val_inputs::AbstractArray{<:Real,2}, test_inputs::AbstractArray{<:Real,2};
    cont_cols::AbstractVector{Int} = [1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14],
    key_col::Int = 6, 
    cat_cols::Int = 12,
    bool_col::Int = 9,
    cat_classes::AbstractArray{<:Any, 1} = [4, 3, 1, 5, 0])

    train_cont = train_inputs[:, cont_cols] # Matrix of continuous variables -> z score normalization

    mean_train = mean(train_cont, dims=1)
    std_train = std(train_cont, dims=1)

    # Training normalization
    norm_cont_train = normalizeZeroMean(train_cont) # Normalize training set z score
    norm_cyclic_train = normalizeCyclic(train_inputs[:, key_col], 12)
    norm_cat_train = oneHotEncoding(train_inputs[:, cat_cols], cat_classes)
    bool_train = train_inputs[:, bool_col]
    final_train = Float32.(hcat(norm_cont_train, norm_cyclic_train, norm_cat_train, bool_train))

    # Validation normalization
    if size(val_inputs, 1) > 0

        norm_cont_val = normalizeZeroMean(val_inputs[:, cont_cols], (mean_train, std_train)) # Normalize validation set z score
        norm_cyclic_val = normalizeCyclic(val_inputs[:, key_col], 12)
        norm_cat_val = oneHotEncoding(val_inputs[:, cat_cols], cat_classes)
        bool_val = val_inputs[:, bool_col]
        final_val = Float32.(hcat(norm_cont_val, norm_cyclic_val, norm_cat_val, bool_val))

    else
        final_val = Matrix{Float32}(undef, 0, 0)
    end

    # Test normalization
    norm_cont_test = normalizeZeroMean(test_inputs[:, cont_cols], (mean_train, std_train)) # Normalize test set z score
    norm_cyclic_test = normalizeCyclic(test_inputs[:, key_col], 12)
    norm_cat_test = oneHotEncoding(test_inputs[:, cat_cols], cat_classes)    
    bool_test = test_inputs[:, bool_col]
    final_test = Float32.(hcat(norm_cont_test, norm_cyclic_test, norm_cat_test, bool_test))

    return final_train, final_val, final_test

end


"""
    ANNCrossValidation(topology, dataset, crossValidationIndices; numExecutions, transferFunctions, maxEpochs, learningRate, validationRatio, maxEpochsVal)

Perform k-fold cross-validation for Artificial Neural Network classification with multiple executions per fold.

Trains and evaluates an ANN classifier using cross-validation with optional validation set for early stopping.
Each fold is executed multiple times to provide robust statistical estimates of performance metrics.

# Arguments
- `topology`: Hidden layer architecture as array of integers
- `dataset`: Tuple of (inputs, categorical_targets)
- `crossValidationIndices`: Array indicating fold assignment for each sample
- `numExecutions`: Number of executions per fold for statistical robustness (default: 50)
- `transferFunctions`: Activation functions for each layer (default: σ)
- `maxEpochs`: Maximum training epochs (default: 1000)
- `minLoss`: Minimum loss threshold for early stopping (default: 0.0)
- `learningRate`: Learning rate for optimizer (default: 0.01)
- `validationRatio`: Proportion of training data for validation (default: 0)
- `maxEpochsVal`: Maximum epochs without improvement for early stopping (default: 20)

# Returns
- `Tuple`: ((mean_accuracy, std_accuracy), (mean_error_rate, std_error_rate), 
           (mean_recall, std_recall), (mean_specificity, std_specificity),
           (mean_precision, std_precision), (mean_npv, std_npv), 
           (mean_f1, std_f1), confusion_matrix, accuracy_vector)
"""
function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20) 

    inputs, targets = dataset # Decompose dataset
    classes = unique(targets) # Calculate classes
    one_hot = oneHotEncoding(targets, classes) # OneHot encode classes

    folds = maximum(crossValidationIndices) # Calculate number of folds

    accuracy = Float64[] # Accuracy vector
    fail_rate = Float64[] # Error rate vector
    recall = Float64[] # Sensitivity (recall) vector
    especificity = Float64[] # Specificity vector
    precision = Float64[] # PPV (precision) vector
    npv = Float64[] # NPV vector
    f1 = Float64[] # F1 score vector
    confussion_matrix = zeros(length(classes), length(classes)) # Initialize confusion matrix

    for fold in 1:folds

        # Extract training and test data according to folds
        train_inputs = inputs[findall(crossValidationIndices .!= fold), :]
        train_targets = one_hot[findall(crossValidationIndices .!= fold), :]
        test_inputs = inputs[findall(crossValidationIndices .== fold), :]
        test_targets = one_hot[findall(crossValidationIndices .== fold), :]

        validation_inputs = Matrix{Float32}(undef, 0, size(train_inputs, 2)) # Initialize validation inputs as empty matrix
        validation_targets = Matrix{Bool}(undef, 0, size(train_targets, 2)) # Initialize validation targets as empty matrix

        if validationRatio > 0 # In case we have validation

            v_ratio = validationRatio * (folds/(folds-1)) # Calculate adapted validation ratio

            trainIndices, validationIndices = holdOut(size(train_inputs, 1), v_ratio) # Calculate validation indices
            validation_inputs = train_inputs[validationIndices, :] # Extract validation inputs
            validation_targets = train_targets[validationIndices, :] # Extract validation targets
            train_inputs = train_inputs[trainIndices, :] # Extract training inputs
            train_targets = train_targets[trainIndices, :] # Extract training targets
        end

        # Normalize fold
        norm_train, norm_val, norm_test = normalizeFold(train_inputs, validation_inputs, test_inputs)

        # Create new vectors for metrics of each epoch of each fold
        acc_fold = []
        fail_rate_fold = []
        recall_fold = []
        especificity_fold = []
        precision_fold = []
        npv_fold = []
        f1_fold = []
        cnf_matrix_fold = Array{Float64}(undef, length(classes), length(classes), numExecutions)

        # Train fold
        for i in 1:numExecutions

            # Train ANN
            ann, _, _, _ = trainClassANN(
                topology, 
                (norm_train, train_targets);
                validationDataset=(norm_val, validation_targets),
                testDataset=(norm_test, test_targets),
                transferFunctions=transferFunctions, 
                maxEpochs=maxEpochs, 
                minLoss=minLoss, 
                learningRate=learningRate)

            # Calculate metrics
            metrics = confusionMatrix(ann(norm_test')', test_targets)

            # Add metrics to record
            push!(acc_fold, metrics[1])
            push!(fail_rate_fold, metrics[2])
            push!(recall_fold, metrics[3])
            push!(especificity_fold, metrics[4])
            push!(precision_fold, metrics[5])
            push!(npv_fold, metrics[6])
            push!(f1_fold, metrics[7])
            cnf_matrix_fold[:, :, i] = metrics[8]

        end

        # Calculate metrics for each fold
        push!(accuracy, mean(acc_fold))
        push!(fail_rate, mean(fail_rate_fold))
        push!(recall, mean(recall_fold))
        push!(especificity, mean(especificity_fold))
        push!(precision, mean(precision_fold))
        push!(npv, mean(npv_fold))
        push!(f1, mean(f1_fold))
        confussion_matrix += dropdims(mean(cnf_matrix_fold, dims=3), dims=3)

    end
    
    # Return mean and standard deviation of metrics
    return ((mean(accuracy), std(accuracy)), (mean(fail_rate), std(fail_rate)), (mean(recall), std(recall)), (mean(especificity), std(especificity)), (mean(precision), std(precision)), (mean(npv), std(npv)), (mean(f1), std(f1)), confussion_matrix, accuracy)

end



"""
    modelCrossValidation(modelType, modelHyperparameters, dataset, crossValidationIndices; normalize=true)

Perform k-fold cross-validation for multiple classification models (ANN, SVM, Decision Tree, kNN, DoME).

Trains and evaluates the specified classifier using cross-validation. Supports optional input normalization and 
returns a set of statistical performance metrics.

# Arguments
- `modelType::Symbol`: Type of model to train (`:ANN`, `:SVC`, `:DecisionTreeClassifier`, `:KNeighborsClassifier`, `:DoME`)
- `modelHyperparameters::Dict`: Dictionary with model-specific hyperparameters
- `dataset::Tuple{Array{<:Real,2}, Array{<:Any,1}}`: Tuple containing input features and target labels
- `crossValidationIndices::Array{Int64,1}`: Array indicating fold assignment for each sample
- `normalize::Bool`: Whether to apply normalization to input features (default: true)

# Returns
- `Tuple`: 
    - `(mean_accuracy, std_accuracy)`
    - `(mean_error_rate, std_error_rate)`
    - `(mean_recall, std_recall)`
    - `(mean_specificity, std_specificity)`
    - `(mean_precision, std_precision)`
    - `(mean_npv, std_npv)`
    - `(mean_f1, std_f1)`
    - `confusion_matrix::Matrix`: Aggregated confusion matrix across folds
    - `accuracy_vector::Vector`: Accuracy per fold
"""


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1}; normalize::Bool = true)

    # Normalize keys: allows strings and symbols
    function get_param(dict, key)
        return get(dict, key, get(dict, Symbol(key), nothing))
    end

    if modelType == :ANN
        # Make sure topology is included
        topology = get_param(modelHyperparameters, "topology")
        @assert(topology !== nothing, "Parameter 'topology' is required for ANN")
        @assert(isa(topology, AbstractArray{<:Int,1}), "topology must be AbstractArray{<:Int,1}")

        # Create new dictionary for optional params
        ann_params = Dict{Symbol, Any}()
        
        # Arguments list with expected data type
        param_types = Dict(
            :numExecutions => Int,
            :transferFunctions => AbstractArray{<:Function,1},
            :maxEpochs => Int,
            :minLoss => Real,
            :learningRate => Real,
            :validationRatio => Real,
            :maxEpochsVal => Int
        )

        # Validate and add optional params
        for (param, param_type) in param_types
            value = get_param(modelHyperparameters, param)
            if value !== nothing
                @assert isa(value, param_type) "$(param) must be of type $(param_type)"
                ann_params[param] = value
            end
        end

        # Call ANNCrossValidation with validated args
        return ANNCrossValidation(
            topology, 
            dataset, 
            crossValidationIndices; 
            pairs(ann_params)...
        )

    else
        inputs, targets = dataset
        targets = string.(targets) # convert to string
        classes = unique(targets)
        folds = maximum(crossValidationIndices)

        accuracy = Float64[]
        fail_rate = Float64[]
        recall = Float64[]
        especificity = Float64[]
        precision = Float64[]
        npv = Float64[]
        f1 = Float64[]
        confussion_matrix = zeros(length(classes), length(classes))

        for fold in 1:folds
            train_inputs = inputs[findall(crossValidationIndices .!= fold), :]
            train_targets = targets[findall(crossValidationIndices .!= fold), :]
            test_inputs = inputs[findall(crossValidationIndices .== fold), :]
            test_targets = targets[findall(crossValidationIndices .== fold), :]
            model_output = [] # Initialize empty output vector

            if normalize == true
                norm_train_inputs, _, norm_test_inputs = normalizeFold(train_inputs, Matrix{Float32}(undef, 0, 0), test_inputs)
            else 
                norm_train_inputs, _, norm_test_inputs = train_inputs, Matrix{Float32}(undef, 0, 0), test_inputs
            end

            if modelType == :DoME
                maximumNodes = get_param(modelHyperparameters, "maximumNodes")
                @assert(maximumNodes !== nothing, "Parameter 'maximumNodes' is required for DoME")
                @assert(isa(maximumNodes, Int), "maximumNodes must be an Int")

                model_output = trainClassDoME((train_inputs, train_targets[:]), test_inputs, maximumNodes)

            elseif modelType == :SVC

                svm_params = Dict{Symbol, Any}()

                C = get_param(modelHyperparameters, "C")
                @assert(C !== nothing, "Parameter 'C' is required for SVM")
                @assert(isa(C, Real), "C must be a Real")

                kernel = get_param(modelHyperparameters, "kernel")
                @assert(kernel !== nothing, "Parameter 'kernel' is required for SVM")
                @assert(isa(kernel, String), "kernel must be String")

                kernel_type = lowercase(kernel)
                kernel_map = Dict(
                    "linear" => LIBSVM.Kernel.Linear,
                    "rbf" => LIBSVM.Kernel.RadialBasis,
                    "radialbasis" => LIBSVM.Kernel.RadialBasis,
                    "sigmoid" => LIBSVM.Kernel.Sigmoid,
                    "poly" => LIBSVM.Kernel.Polynomial,
                    "polynomial" => LIBSVM.Kernel.Polynomial
                )

                @assert(haskey(kernel_map, kernel_type), "Kernel not supported: $(kernel)")

                kernel = kernel_map[kernel_type]

                if kernel in [LIBSVM.Kernel.RadialBasis, LIBSVM.Kernel.Sigmoid, LIBSVM.Kernel.Polynomial]
                    gamma = get_param(modelHyperparameters, "gamma")
                    @assert(gamma !== nothing, "Parameter 'gamma' required for $(kernel_type) SVM")
                    @assert(isa(gamma, Real), "gamma must be a Real")
                    svm_params[:gamma] = Float64(gamma)
                end

                if kernel in [LIBSVM.Kernel.Sigmoid, LIBSVM.Kernel.Polynomial]
                    coef0 = get_param(modelHyperparameters, "coef0")
                    @assert(coef0 !== nothing, "Parameter 'coef0' required for $(kernel_type) SVM")
                    @assert(isa(coef0, Real), "coef0 must be a Real")
                    svm_params[:coef0] = Float64(coef0)
                end

                if kernel == LIBSVM.Kernel.Polynomial
                    degree = get_param(modelHyperparameters, "degree")
                    @assert(degree !== nothing, "Parameter 'degree' required for Polynomial SVM")
                    @assert(isa(degree, Int), "degree must be an Int")
                    svm_params[:degree] = Int32(degree)
                end

                model = SVMClassifier(kernel=kernel, cost=Float64(C); svm_params...)
                mach = machine(model, MLJ.table(norm_train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                model_output = MLJ.predict(mach, MLJ.table(norm_test_inputs))

            elseif modelType == :DecisionTreeClassifier

                max_depth = get_param(modelHyperparameters, "max_depth")
                @assert(max_depth !== nothing, "max_depth required for DecisionTree")

                rng = rng = Random.MersenneTwister(1) 
                
                model = DTClassifier(max_depth=max_depth, rng=rng)
                mach = machine(model, MLJ.table(norm_train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                output = MLJ.predict(mach, MLJ.table(norm_test_inputs))
                model_output = mode.(output)

            elseif modelType == :KNeighborsClassifier
                
                n_neighbors = get_param(modelHyperparameters, "n_neighbors")
                @assert(n_neighbors !== nothing, "Parameter 'n_neighbors' required for KNN")
                
                model = kNNClassifier(K = n_neighbors)
                mach = machine(model, MLJ.table(norm_train_inputs), categorical(train_targets[:]))
                MLJ.fit!(mach, verbosity=0)
                output = MLJ.predict(mach, MLJ.table(norm_test_inputs))
                model_output = mode.(output)
            
            else
                println("Model not supported: $modelType, check the function parameters")
                return
            end

            # Make sure we can calculate metrics
            @assert length(model_output) == length(test_targets[:]) "Dimensions of model_output and test_targets do not match"

            # Calculate fold's metrics
            metrics = confusionMatrix(model_output, test_targets[:], classes)
            push!(accuracy, metrics[1])
            push!(fail_rate, metrics[2])
            push!(recall, metrics[3])
            push!(especificity, metrics[4])
            push!(precision, metrics[5])
            push!(npv, metrics[6])
            push!(f1, metrics[7])
            confussion_matrix += metrics[8]
        end


        return (
            (mean(accuracy), std(accuracy)),
            (mean(fail_rate), std(fail_rate)),
            (mean(recall), std(recall)),
            (mean(especificity), std(especificity)),
            (mean(precision), std(precision)),
            (mean(npv), std(npv)),
            (mean(f1), std(f1)),
            confussion_matrix,
            accuracy
           )
        
    end
end


"""
Load a vector of indices from a CSV file.

Args:
    filepath (String): Path to the CSV file containing the indices.  
        The file is expected to have no header, and a column will be auto-assigned as `:Column1`.

Returns:
    Vector{Int64}: A vector containing the indices.
"""
function load_indices_vector(filepath::String)
    # Read without header; column is automatically named :Column1
    df = CSV.read(filepath, DataFrame)    
    return Int64.(df.fold)
end
