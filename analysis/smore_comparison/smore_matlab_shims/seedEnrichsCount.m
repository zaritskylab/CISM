function [zoopsCnt, totalCnt] = seedEnrichsCount(seedsWMax, psSeedIndex)
% CPU equivalent of SMORE's seedEnrichsCount.m GPU membership step.

numSeeds = length(psSeedIndex);
[~, indx] = ismember(seedsWMax.pSeedsIndex, psSeedIndex);
seedsSites = cell(numSeeds, 1);
seedsSites(indx) = seedsWMax.pSiteCell;
seedsWeights = cell(numSeeds, 1);
seedsWeights(indx) = seedsWMax.pWeightCell;
pWeights = vertcat(seedsWeights{:});

seedsCounts = zeros(length(seedsSites), 1);
for i = 1:length(seedsWeights)
    seedsCounts(i) = sum(seedsWeights{i});
end
W = size(seedsWMax.seeds, 2);
[~, seedIndex] = ismember(psSeedIndex, seedsWMax.seedsIndex);
totalCnt = seedsWMax.weights(seedIndex, :);
sitesFlag = repelem((1:length(seedsCounts)).', seedsCounts, 1);
seedsSites = vertcat(seedsSites{:});
nodes = seedsSites.';
nodes = nodes(:);
if length(nodes) > 5e6
    warning('Large seedEnrichsCount input: %d nodes', length(nodes));
end

siteMask = getZNICIM(nodes, ones(W, 1));
siteMask = sum(reshape(siteMask, W, [])).';
siteMask = siteMask > 0;
sitesFlag = sitesFlag(siteMask);
pWeights = pWeights(siteMask);
[sitesUnique, firstIndices, groupIndices] = uniqueSorted(sitesFlag);
counts = zeros(length(firstIndices), 1);
if ~isempty(groupIndices)
    summed = accumarray(groupIndices, pWeights);
    counts(1:length(summed)) = summed;
end
zoopsCnt = zeros(numSeeds, 1);
zoopsCnt(sitesUnique) = counts;
end
