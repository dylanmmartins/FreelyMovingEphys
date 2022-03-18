%% generate kilosort maps

%% DB P128-6

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*150 ones(1,32)*300 ones(1,32)*450];

%%% y coordinates for each site
[775:-25:0 775:-25:0 775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2 ones(1,32)*3 ones(1,32)*4];

%%% channel map
1:128;

%% DB P64-3

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*250];

%%% y coordinates for each site
[775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64

%% DB P64-8

%%% x coordinates for each site
[repmat([21,0],1,16) repmat([271,250],1,16)]

%%% y coordinates for each site
[387.5:-12.5:0 387.5:-12.5:0]

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64


%% NN H64-LP

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*250];

%%% y coordinates for each site
[775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64


%% NN 16ch

%%% x coordinates for each site
zeros(1,16)

%%% y coordinates for each site
375:-25:0

%%% shank index
ones(1,16)

%%% channel map
1:16


%% DB P64-10-D (Janus double sided)

%%% sites are 11um wide, top sites are 52.5um apart, bottom are 11um apart
%%% (center to center)
%%% V-shaped shanks (4 sides total)
%%% two physical shanks are 250um apart
P = 250; %%% pitch a.k.a. distance between shanks

%%% x coordinates for each site
first = linspace(0,20.75,16);
second = linspace(52.5,31.75,16);
comb = [first;second];
comb_1 = comb(:)';

first = linspace(0+P,20.75+P,16);
second = linspace(52.5+P,31.75+P,16);
comb = [first;second];
comb_2 = comb(:)';

second = linspace(0,20.75,16);
first = linspace(52.5,31.75,16);
comb = [first;second];
comb_3 = comb(:)';

second = linspace(0+P,20.75+P,16);
first = linspace(52.5+P,31.75+P,16);
comb = [first;second];
comb_4 = comb(:)';

[comb_1 comb_2 comb_3 comb_4];

[0	52.5000000000000	1.38333333333333	51.1166666666667	2.76666666666667	49.7333333333333	4.15000000000000	48.3500000000000	5.53333333333333	46.9666666666667	6.91666666666667	45.5833333333333	8.30000000000000	44.2000000000000	9.68333333333333	42.8166666666667	11.0666666666667	41.4333333333333	12.4500000000000	40.0500000000000	13.8333333333333	38.6666666666667	15.2166666666667	37.2833333333333	16.6000000000000	35.9000000000000	17.9833333333333	34.5166666666667	19.3666666666667	33.1333333333333	20.7500000000000	31.7500000000000	250	302.500000000000	251.383333333333	301.116666666667	252.766666666667	299.733333333333	254.150000000000	298.350000000000	255.533333333333	296.966666666667	256.916666666667	295.583333333333	258.300000000000	294.200000000000	259.683333333333	292.816666666667	261.066666666667	291.433333333333	262.450000000000	290.050000000000	263.833333333333	288.666666666667	265.216666666667	287.283333333333	266.600000000000	285.900000000000	267.983333333333	284.516666666667	269.366666666667	283.133333333333	270.750000000000	281.750000000000	52.5000000000000	0	51.1166666666667	1.38333333333333	49.7333333333333	2.76666666666667	48.3500000000000	4.15000000000000	46.9666666666667	5.53333333333333	45.5833333333333	6.91666666666667	44.2000000000000	8.30000000000000	42.8166666666667	9.68333333333333	41.4333333333333	11.0666666666667	40.0500000000000	12.4500000000000	38.6666666666667	13.8333333333333	37.2833333333333	15.2166666666667	35.9000000000000	16.6000000000000	34.5166666666667	17.9833333333333	33.1333333333333	19.3666666666667	31.7500000000000	20.7500000000000	302.500000000000	250	301.116666666667	251.383333333333	299.733333333333	252.766666666667	298.350000000000	254.150000000000	296.966666666667	255.533333333333	295.583333333333	256.916666666667	294.200000000000	258.300000000000	292.816666666667	259.683333333333	291.433333333333	261.066666666667	290.050000000000	262.450000000000	288.666666666667	263.833333333333	287.283333333333	265.216666666667	285.900000000000	266.600000000000	284.516666666667	267.983333333333	283.133333333333	269.366666666667	281.750000000000	270.750000000000]

%%% y coordinates for each site
[775:-50:25 750:-50:0 775:-50:25 750:-50:0 775:-50:25 750:-50:0 775:-50:25 750:-50:0];

%%% shank index
[ones(1,32) ones(1,32)*2 ones(1,32)*3 ones(1,32)*4];

%%% channel map
1:128;
