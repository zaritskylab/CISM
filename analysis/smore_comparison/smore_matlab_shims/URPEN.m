function [nodeList, listLength, pathSections, subPathWC] = URPEN(gStruct, options)
% Serial form of upstream URPEN.m for deterministic seeded random sampling.

numVertex = length(gStruct.neighs);
subPathCell = cell(numVertex, 1);
subPathWC = cell(numVertex, 1);
pathSections = cell(numVertex, 1);
listLength = zeros(numVertex, 1);
for v = 1:numVertex
    extension = gStruct.neighs{v};
    extensionWeights = gStruct.weights{v};
    [subPaths, pathWeights] = extendSubupath( ...
        v, extension, extensionWeights, gStruct, [], 1, [], options);
    subPathCell{v} = subPaths;
    subPathWC{v} = pathWeights;
    pathSections{v} = repelem(gStruct.labels(v, 2), size(subPaths, 1), 1);
    listLength(v) = size(subPaths, 1);
end
paths = vertcat(subPathCell{:});
subPathWC = vertcat(subPathWC{:});
nodeList = num2cell(paths, 2);
pathSections = vertcat(pathSections{:});
end


function [subPaths, pathWeights] = extendSubupath( ...
    currentPath, extension, extensionWeights, graphStruct, ...
    subPaths, weightSoFar, pathWeights, options)

k = options.k;
pdv = options.pdv;
if length(currentPath) == k
    if currentPath(k) > currentPath(1)
        if ~options.isChRadial || checkRadial(currentPath, graphStruct)
            subPaths = [subPaths; currentPath];
            pathWeights = [pathWeights; weightSoFar];
        end
    end
else
    while ~isempty(extension)
        nextNode = extension(end);
        extension(end) = [];
        edgeWeight = extensionWeights(end);
        extensionWeights(end) = [];
        if rand < pdv(length(currentPath) + 1)
            nextWeight = edgeWeight * weightSoFar;
            neighbors = graphStruct.neighs{nextNode};
            neighborWeights = graphStruct.weights{nextNode};
            if options.isChRadial
                priorNeighbors = graphStruct.neighs(currentPath);
                priorNeighbors = vertcat(priorNeighbors{:});
                priorNeighbors = unique(priorNeighbors);
            else
                priorNeighbors = [];
            end
            [nextExtension, indices] = setdiff( ...
                neighbors, [currentPath(:); priorNeighbors]);
            nextExtensionWeights = neighborWeights(indices);
            [subPaths, pathWeights] = extendSubupath( ...
                [currentPath, nextNode], nextExtension, nextExtensionWeights, ...
                graphStruct, subPaths, nextWeight, pathWeights, options);
        end
    end
end
end
