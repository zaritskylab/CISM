function [PWMScore, PWMScoreLetter] = scoreWords(seeds, PWMS, scrSpecs)
% CPU equivalent of SMORE's scoreWords.m (upstream uses gpuArray unconditionally).

order = scrSpecs.mkvOrder;
[numCells, W] = size(PWMS);
PWMScoreLetter = -10 * ones(size(seeds));
wordsErased = any(seeds == 0, 2);
if any(wordsErased)
    seeds = seeds(~wordsErased, :);
end

wordsC = seeds + (0:W-1) * numCells;
wordScores = PWMS(wordsC(:));
letterScores = reshape(wordScores, size(seeds, 1), W);
if any(wordsErased)
    PWMScoreLetter(~wordsErased, :) = letterScores;
else
    PWMScoreLetter = letterScores;
end
PWMScore = sum(PWMScoreLetter, 2);

if order > 0
    scoreBack = getScoreBack(seeds, scrSpecs);
    PWMScoreLetter(~wordsErased, :) = PWMScoreLetter(~wordsErased, :) - scoreBack;
    PWMScore = sum(PWMScoreLetter, 2);
end
end
