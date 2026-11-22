function run_smore_matched_fov(input_csv, output_dir, smore_root, path_probability, max_motifs, n_train, n_score, graph_mode, random_seed, backend, is_enrich)
% Run the official SMORE discovery functions on one matched melanoma FOV.

if nargin < 4
    path_probability = 1;
end
if nargin < 5
    max_motifs = 5;
end
if nargin < 6
    n_train = 50;
end
if nargin < 7
    n_score = 10;
end
if nargin < 8
    graph_mode = 'delaunay';
end
if nargin < 9
    random_seed = 1240;
end
if nargin < 10
    backend = 'cpu';
end
if nargin < 11
    is_enrich = true;
end
if path_probability <= 0 || path_probability > 1
    error('SMORE:BadProbability', 'Path probability must be in (0, 1]');
end
if ~ismember(string(graph_mode), ["delaunay", "epsilon"])
    error('SMORE:BadGraphMode', 'Graph mode must be delaunay or epsilon');
end
addpath(genpath(fullfile(smore_root, 'func')));
if strcmpi(backend, 'gpu')
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
    fprintf('URPEN: %s\nscoreWords: %s\nseedEnrichsCount: %s\n', ...
        which('URPEN'), which('scoreWords'), which('seedEnrichsCount'));
    if isempty(gcp('nocreate'))
        parpool('local', 4);
    end
elseif strcmpi(backend, 'cpu')
    addpath(fullfile(fileparts(mfilename('fullpath')), 'smore_matlab_shims'));
else
    error('SMORE:BadBackend', 'Backend must be cpu or gpu');
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end
[~, fov] = fileparts(input_csv);
output_mat = fullfile(output_dir, strcat(fov, '.mat'));
output_csv = fullfile(output_dir, strcat(fov, '_motifs.csv'));
output_meta = fullfile(output_dir, strcat(fov, '_metadata.csv'));

cg.isJClusterID = false;
cg.isNoiseCTS = true;
cg.plotGraph = false;
cg.isUniformWeight = true;
cg.isRandom = false;
cg.randCoords = false;
cg.gMode = graph_mode;
cg.nNeighs = 6;
cg.rEps = 100;
cg.rEpsilon = 0;
cg.xyNoiseStd = 0;
cg.ND = 0;
cg.SIDSel = [];
cg.cellTypeName = 'cellType';
cg.iSelROI = false;
cg.xyLTh = [0, 0];
cg.xyHTh = [inf, inf];
cg.isDiscardCellType = false;
cg.isTileSIDs = false;

[G, graph_struct, ~, cell_types, ~, ~, ~] = creatAGraph(input_csv, cg);
if numnodes(G) < 4
    error('SMORE:TooFewCells', '%s has fewer than four graph cells', fov);
end

rng(random_seed);
opts.pdv = [1, repmat(path_probability^(1/3), 1, 3)];
opts.k = 4;
opts.isChRadial = true;
opts.disObvs = false;
opts.cPTypes = G.Nodes.label(:, 1);
[paths, ~, sections, weights] = URPEN(graph_struct, opts);
cell_types_primary = G.Nodes.label(:, 1);

shuffle.fixedTypes = 0;
shuffle.xyCoordinates = G.Nodes.Coordinates;
shuffle.nearNeighs = [];
shuffle.partitionInd = [];
shuffle.fixedNodes = [];
shuffle.shuffleMode = 'shuffle';
shuffle.rEpsilon = 0;
shuffle.isSectShuffle = true;
shuffle.numClusters = 1;
shuffle.cSections = G.Nodes.label(:, 2).';
shuffle.numShuffle = 1;
rng(random_seed + 521);
cell_types_control = getShuffleTypes(cell_types_primary.', shuffle);

gen.hFrac = 0;
gen.mkvOrder = 0;
gen.rvp = true;
gen.isSectHold = true;
gen.isHExclsv = true;
gen.numNodes = numel(cell_types_primary);
gen.disObvs = false;
gen.numGRs = 1;
[positive, ~, positive_types, positive_weights] = generateSeqs( ...
    paths, weights, sections, cell_types_primary.', gen);
[~, ~, negative_types, negative_weights] = generateSeqs( ...
    paths, weights, sections, cell_types_control, gen);

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
alphabet = alphabet(1:numel(cell_types));
rng(random_seed + 521);
[motifs, stop_reason] = mtSmore( ...
    cPTypes=positive_types, cNTypes=negative_types, ...
    cPHTypes=cell_types_primary, cNHTypes=cell_types_control(:), ...
    pSeq=positive, posWeight=positive_weights, negWeight=negative_weights, ...
    rvp=true, mkvOrder=0, wMin=4, wMax=4, ...
    nmotifs=max_motifs, threshold=0.05, patience=3, ...
    alphabet=alphabet, fixedTypes=0, shConfig=shuffle, ...
    shuffleMode='shuffle', nRefIter=20, gTrainNum=n_train, ...
    isEnrich=is_enrich, scIterMax=n_score, NREF=4, NEVAL=25, ...
    cellTypesStr=string(cell_types));

save(output_mat, 'motifs', 'cell_types', 'stop_reason', 'input_csv', ...
    'smore_root', 'path_probability', 'max_motifs', 'n_train', 'n_score', ...
    'graph_mode', 'random_seed', 'backend', 'is_enrich', '-v7.3');
rows = cell(numel(motifs), 14);
for idx = 1:numel(motifs)
    motif = motifs{idx};
    consensus = strjoin(string(cell_types(motif.cSeed)), '-');
    observed_count = sum(motif.pnsCPV(:, 1));
    rows(idx, :) = {string(fov), idx, motif.testPvalue, ...
        exp(motif.testPvalue), motif.testPvalue < log(0.05), ...
        consensus, observed_count, numel(paths), ...
        observed_count / numel(paths), path_probability, ...
        n_train, n_score, string(graph_mode), jsonencode(motif.PWM)};
end
catalog = cell2table(rows, 'VariableNames', ...
    {'fov', 'motif_number', 'log_p_value', 'p_value', 'significant', ...
     'consensus_cell_types', 'observed_znic_count', 'sampled_radial_paths', ...
     'sampled_frequency', 'path_probability', 'n_train', 'n_score', ...
     'graph_mode', 'pwm_json'});
writetable(catalog, output_csv);
metadata = table(string(fov), numnodes(G), numel(paths), ...
    path_probability, max_motifs, n_train, n_score, string(graph_mode), ...
    random_seed, string(backend), is_enrich, ...
    'VariableNames', {'fov', 'graph_cells', 'sampled_radial_paths', ...
    'path_probability', 'max_motifs', 'n_train', 'n_score', 'graph_mode', ...
    'random_seed', 'backend', 'is_enrich'});
writetable(metadata, output_meta);
fprintf('%s: %d radial paths, %d output motifs, %d significant\n', ...
    fov, numel(paths), numel(motifs), sum(catalog.significant));
end
