%% Initialization routine for 3D turbulence
%% Use Erlebacher's method

clear;close all;

procDim_y = 2;
procDim_x = 2;
procDim_z = 2;

N = 256;
Np = N*N*N;	
%% Spectrum constant: Don't change

Mt = 0.488; Re  = 175; k0 = 4;
A = (sqrt(2/pi)*(16/3)*Mt^2/k0^5);% for energy spectrum, constant
enrgySpec=@(wavenumber) A*wavenumber.^4.*exp(-2*(wavenumber/k0).^2)*Np^2;

kmin = -floor(0.5*N)+1;
kmax = floor(0.5*N);
krange = kmin:kmax;

I = sqrt(-1);		%complex i

%% Generate random data
fprintf('uks begin...\n');
u = rand(N,N,N);  uks = fft3(u); clear u; fprintf('uks done...\n');
fprintf('vks begin...\n');
v = rand(N,N,N);  vks = fft3(v); clear v; fprintf('vks done...\n');
fprintf('wks begin...\n');
w = rand(N,N,N);  wks = fft3(w); clear w; fprintf('wks done...\n');

%% Make velocity field divergence free
kfft = [0:kmax kmin:-1];
[k1,k2,k3] = meshgrid(kfft,kfft,kfft);
ksq     = max(k1.^2 + k2.^2 + k3.^2,1.e-14);

kdotvel = k1.*uks + k2.*vks + k3.*wks;
fprintf('Imposing Incompressibility...\n');
uks = uks - kdotvel.*k1./ksq;
vks = vks - kdotvel.*k2./ksq;
wks = wks - kdotvel.*k3./ksq;
fprintf('Imposing desired spectrum...\n');

%% Compute obtained spectrum
for ix = 1:N
 for jy = 1:N
   for kz = 1:N
       kk=sqrt(kfft(ix)*kfft(ix)+kfft(jy)*kfft(jy)+kfft(kz)*kfft(kz));
       if(kk>=kcmin && kk<=kcmax)
           ee=enrgySpec(kk);
           ee=ee/(4*pi*kk*kk);
           umag2=uks(ix,jy,kz).^2+vks(ix,jy,kz).^2+wks(ix,jy,kz).^2;
           uks(ix,jy,kz)=uks(ix,jy,kz)*sqrt(ee/umag2);
           vks(ix,jy,kz)=vks(ix,jy,kz)*sqrt(ee/umag2);
           wks(ix,jy,kz)=wks(ix,jy,kz)*sqrt(ee/umag2);
       else
           uks(ix,jy,kz)=0;vks(ix,jy,kz)=0;wks(ix,jy,kz)=0;
       end
   end
 end
end

uks=uks/Np; vks=vks/Np; wks=wks/Np;

fprintf('IFFT...\n');
u = ifft3(uks); u = real(u); clear uks; fprintf('u ifft done...\n');
v = ifft3(vks); v = real(v); clear vks; fprintf('v ifft done...\n');
w = ifft3(wks); w = real(w); clear wks; fprintf('w ifft done...\n');

fprintf('Field obtained. \n');

ek0 = 0.5*sum(sum(sum(u.^2 + v.^2 + w.^2))) / (N^3);


%% Imposing Mach and Re
gam = 7/5;
rho0 = 1;
T0 = 300;
R  = 287;
p0 = rho0*R*T0;

M = sqrt(2*ek0) / sqrt(gam*p0/rho0);
eta = Mt/M;

u = u*eta;
v = v*eta;
w = w*eta;

ek0 = 0.5*sum(sum(sum(u.^2 + v.^2 + w.^2))) / (N^3);
Mt = sqrt(2*ek0) / sqrt(gam*p0/rho0);

ek0 = 0.5*sum(sum(sum(u.^2 + v.^2 + w.^2))) / (N^3);

i = 1:N; ip = [2:N 1]; im = [N 1:N-1];
j = 1:N; jp = [2:N 1]; jm = [N 1:N-1];
k = 1:N; kp = [2:N 1]; km = [N 1:N-1];

dx = 2*pi/N;

dudx = .5*(u(i,jp,k) - u(i,jm,k))/dx;
dvdy = .5*(v(ip,j,k) - v(im,j,k))/dx;
dwdz = .5*(w(i,j,kp) - w(i,j,km))/dx;
dudy = .5*(u(ip,j,k) - u(im,j,k))/dx;
dudz = .5*(u(i,j,kp) - u(i,j,km))/dx;
dvdx = .5*(v(i,jp,k) - v(i,jm,k))/dx;
dvdz = .5*(v(i,j,kp) - v(i,j,km))/dx;
dwdy = .5*(w(ip,j,k) - w(im,j,k))/dx;
dwdx = .5*(w(i,jp,k) - w(i,jm,k))/dx;

fprintf('derivs done...\n');
SijSij = sum(sum(sum( dudx.^2 + dvdy.^2 + dwdz.^2 + ...
         0.25*(dudy + dvdx).^2 + ...
         0.25*(dudz + dwdx).^2 + ...
         0.25*(dvdz + dwdy).^2 ))) / (N^3);


sqdu_idx_j = sum(sum(sum(dudx.*dudx + dudy.*dudy + dudz.*dudz + dvdx.*dvdx + dvdy.*dvdy + dvdz.*dvdz + dwdx.*dwdx + dwdy.*dwdy + dwdz.*dwdz)))/(N^3);
mu = 2*ek0*sqrt(5/(3*sqdu_idx_j))/Re;

diss0 = (mu/rho0)*sqdu_idx_j;
eddy_time1 = ek0/diss0;
lam = sqrt(10*ek0/sqdu_idx_j);
urms = sqrt(2*ek0/3);
eddy_time = lam/urms;
fprintf('Eddy done...\n');

% ------------------------------------------------------------------------
% Writing Data
% ------------------------------------------------------------------------

fprintf('Writing to the file... \n');

seg1Dy = N/procDim_y;
seg1Dx = N/procDim_x;
seg1Dz = N/procDim_z;

us = zeros(seg1Dy,seg1Dx,seg1Dz);
vs = us; ws = us; ps = us; ds = us;
for i=1:procDim_y
    for j=1:procDim_x
        for k=1:procDim_z     
        us = u((i-1)*seg1Dy+1:(i)*seg1Dy, (j-1)*seg1Dx+1:(j)*seg1Dx, (k-1)*seg1Dz+1:(k)*seg1Dz);
        vs = v((i-1)*seg1Dy+1:(i)*seg1Dy, (j-1)*seg1Dx+1:(j)*seg1Dx, (k-1)*seg1Dz+1:(k)*seg1Dz);
        ws = w((i-1)*seg1Dy+1:(i)*seg1Dy, (j-1)*seg1Dx+1:(j)*seg1Dx, (k-1)*seg1Dz+1:(k)*seg1Dz);
        ds(:, :, :) = rho0;
        ps(:, :, :) = p0;
        File = sprintf('Init%d%d%d', i-1, j-1, k-1);

        fid = fopen(File,'w');
        fwrite(fid, us, 'double');
        fwrite(fid, vs, 'double');
        fwrite(fid, ws, 'double');
        fwrite(fid, ds, 'double');
        fwrite(fid, ps, 'double');
        fclose(fid);
        
        end
    end
end

fprintf('Macro Variables Written ...\n');

File = sprintf('params');
fid = fopen(File,'w');
fwrite(fid, N, 'int');
fwrite(fid, gam, 'double');
fwrite(fid, rho0, 'double');
fwrite(fid, Mt   , 'double');
fwrite(fid, Re   , 'double');
fwrite(fid, mu , 'double');
fwrite(fid, p0 , 'double');
fwrite(fid, eddy_time , 'double');
fclose(fid);       

fprintf('End of Initialization ...\n');


fileID = fopen('simdata.txt','w');
fprintf(fileID,'Mt = %f\n',Mt);
fprintf(fileID,'Re_lam = <u_iu_i> * sqrt(5/(3*<AijAij>)) / nu = urms*lam/nu = %f\n',Re);
fprintf(fileID,'lam = sqrt(5*<uiui>/(AijAij)) = sqrt(10*nu*ek0/eps0) = sqrt(10*ek0/<AijAij>) = %f\n',lam);
fprintf(fileID,'A=%f; k0 = %f \n enrgySpec= A*k.^4.*exp(-2*(k/k0).^2)*Np^2', A, k0);
fprintf(fileID,'nc = %f\n',N);
fprintf(fileID,'T0 = %f\n',T0);
fprintf(fileID,'p0 = %f\n',p0);
fprintf(fileID,'den0 = %f\n',rho0);
fprintf(fileID,'gam = %f\n',gam);
fprintf(fileID,'Mt = %f\n',Mt);
fprintf(fileID,'nu = %f\n',mu);
fprintf(fileID,'PRANDTL NO = 0.7 \n');
fprintf(fileID,'<AijAij> = %f \n', sqdu_idx_j);
fprintf(fileID,'eps0 = <nu*(A_ij A_ij) = %f\n',diss0);
fprintf(fileID,'ek0 = <1/2(u_i*u_i))> = %f\n',ek0);
fprintf(fileID,'urms = %f\n',urms);
fprintf(fileID,'eddy_time = %f\n',eddy_time1);
fprintf(fileID,'Samtaney_eddytime = lam/urms = %f\n',eddy_time);
fprintf(fileID,'kolmogorov length scale = (nu^3 / eps0)^(0.25) = %f\n',(mu^3 / diss0)^(0.25));
fprintf(fileID,'simulation dx = %f \n',  2*pi/N);
fprintf(fileID,'kolmogorov time scale =  %f \n',  (mu/diss0)^(0.5));
fclose(fileID);
