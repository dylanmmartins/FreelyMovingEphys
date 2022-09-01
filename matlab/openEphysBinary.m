%% open ephys binary and check length

binPath = '/path/to/ephys.bin';

%%
openFile = fopen(binPath, 'r');
data = fread(openFile, [chNum Inf], '*uint16');
display(size(data))