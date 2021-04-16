close all
clear all
dbstop if error

%% folder where the data live:
foldername = 'T:\freely_moving_ephys\widefield\';
pathname = foldername;
datapathname = foldername;  
outpathname = foldername;

%start the counter
n = 0;

%% analyzed 020921 PRLP
% n=n+1;
% files(n).subj = 'EE11P11LT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOX\020821_EE11P11LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOX\020821_EE11P11LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOY\020821_EE11P11LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOY\020821_EE11P11LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 020921 PRLP
% n=n+1;
% files(n).subj = 'EE11P11RT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOX\020821_EE11P11RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOX\020821_EE11P11RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOY\020821_EE11P11RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOY\020821_EE11P11RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021421 PRLP
% n=n+1;
% files(n).subj = 'EE11P11TT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOX\020821_EE11P11TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOX\020821_EE11P11TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOY\020821_EE11P11TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOY\020821_EE11P11TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021521 NC NAME INCORRECT IN FOLDER
% n=n+1;
% files(n).subj = 'EE12P1LN';
% files(n).expt = '021521';
% files(n).topox =  '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOX\021521_EE13P2LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOX\021521_EE13P2LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOY\021521_EE13P2LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOY\021521_EE13P2LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021521 NC
% n=n+1;
% files(n).subj = 'EE13P2LT';
% files(n).expt = '021521';
% files(n).topox =  '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOX\021521_EE13P2LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOX\021521_EE13P2LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOY\021521_EE13P2LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOY\021521_EE13P2LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 022221 NC
% n=n+1;
% files(n).subj = 'EE11P12LT';
% files(n).expt = '022221';
% files(n).topox =  '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOX\022221_EE11P12LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOX\022221_EE11P12LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOY\022221_EE11P12LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOY\022221_EE11P12LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 022221 NC
% n=n+1;
% files(n).subj = 'EE12P3LT';
% files(n).expt = '022221';
% files(n).topox =  '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOX\022221_EE12P3LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOX\022221_EE12P3LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOY\022221_EE12P3LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOY\022221_EE12P3LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030121 NC
% n=n+1;
% files(n).subj = 'EE8P7LT';
% files(n).expt = '030121';
% files(n).topox =  '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOX\030121_EE8P7LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOX\030121_EE8P7LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOY\030121_EE8P7LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOY\030121_EE8P7LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';



%% analyzed 030121 NC
% n=n+1;
% files(n).subj = 'EE8P7RT';
% files(n).expt = '030121';
% files(n).topox =  '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOX\030121_EE8P7RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOX\030121_EE8P7RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOY\030121_EE8P7RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOY\030121_EE8P7RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030821 NC 
% n=n+1;
% files(n).subj = 'EE8P8LT';
% files(n).expt = '030821';
% files(n).topox =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030821 NC 
% % n=n+1;
% % files(n).subj = 'EE8P8RT';
% % files(n).expt = '030821';
% % files(n).topox =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOXmaps.mat';
% % files(n).topoxdata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOX';
% % files(n).topoy =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOYmaps.mat';
% % files(n).topoydata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOY';
% % files(n).rignum = 'rig2'; %%% or 'rig1'
% % files(n).monitor = 'land'; %%% for topox and y
% % files(n).label = 'camk2 gc6';
% % files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1RT';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOX\032221_EE14P1RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOX\032221_EE14P1RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOY\032221_EE14P1RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOY\032221_EE14P1RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1LN';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOX\032221_EE14P1LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOX\032221_EE14P1LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOY\032221_EE14P1LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOY\032221_EE14P1LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1RN';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOX\032221_EE14P1RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOX\032221_EE14P1RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOY\032221_EE14P1RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOY\032221_EE14P1RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1TT';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOX\032221_EE14P1TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOX\032221_EE14P1TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOY\032221_EE14P1TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOY\032221_EE14P1TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P1LT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOX\032921_EE14P1LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOX\032921_EE14P1LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOY\032921_EE14P1LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOY\032921_EE14P1LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% % % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P2RT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOX\032921_EE14P2RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOX\032921_EE14P2RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOY\032921_EE14P2RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOY\032921_EE14P2RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P1NN';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOX\032921_EE14P1NN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOX\032921_EE14P1NN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOY\032921_EE14P1NN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOY\032921_EE14P1NN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 032921 EL 
% n=n+1;
% files(n).subj = 'EE14P1TT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOX\032921_EE14P1TT_RIG2_2PMAP_TOPOX2PMAPs.mat';
% files(n).topoxdata = '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOX\032921_EE14P1TT_RIG2_2PMAP_TOPOX';
% files(n).topoy =  '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOY\032921_EE14P1TT_RIG2_2PMAP_TOPOY2PMAPs.mat';
% files(n).topoydata = '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOY\032921_EE14P1TT_RIG2_2PMAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538TT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOX\040521_J538TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOX\040521_J538TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOY\040521_J538TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOY\040521_J538TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538LT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOX\040521_J538LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOX\040521_J538LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOY\040521_J538LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOY\040521_J538LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538RT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOX\040521_J538RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOX\040521_J538RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOY\040521_J538RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOY\040521_J538RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 041321 NC 
% n=n+1;
% files(n).subj = 'J539LT';
% files(n).expt = '041321';
% files(n).topox =  '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOX\041321_J539LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOX\041321_J539LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOY\041321_J539LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOY\041321_J539LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 041321 NC 
% n=n+1;
% files(n).subj = 'J539LN';
% files(n).expt = '041321';
% files(n).topox =  '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOX\041321_J539LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOX\041321_J539LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOY\041321_J539LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOY\041321_J539LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% run the batch analysis (keep this at the bottom of the script
batchDfofMovie


