N = 1000000;
a = rand(N,3);
a = a.*(2*pi);

x = a(:,1);
y = a(:,2);
z = a(:,3);

File = sprintf('unirand');
fid = fopen(File,'w');
fwrite(fid, x, 'double');
fwrite(fid, y, 'double');
fwrite(fid, z, 'double');
fclose(fid);

File = sprintf('particleCount');
fid = fopen(File,'w');
fwrite(fid, N, 'int');
fclose(fid);