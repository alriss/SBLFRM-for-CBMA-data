%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	Code to implement Spatial Bayesian LFRM for CBMA % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cd('/Users/....')     % Set working directory to source code location
clear; clc; 
warning off all;
format compact;
rng(221103);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Read in the CBMA data  % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = dlmread('foci.txt');                         % 3D example: data matrix with columns (study ID, x, y, z)
foci = data(:,2:4);
N = max(data(:, 1)); 				                % Total number of CBMA studies, studies IDs ordered from 1 to N
d = size(data,2) - 1;                               % Number of dimensions (d = 3 for 3D data)
ni = tabulate(data(:,1)); ni = ni(:, 2); 			% Number of foci per study
ID = data(:, 1);                                    % Study ID
data = [data(:, 2:4) ID];                           % Re-organise columns as (x, y, z, study ID)

% -- % Convert MNI coordinates to voxel space % -- %
origin = [90, -126, -72];
data(:, 1) = round((origin(1) - data(:, 1))./2);
data(:, 2) = round((data(:, 2) - origin(2))./2);
data(:, 3) = round((data(:, 3) - origin(3))./2);

% -- % Foci that fall outside [1,91] x [1,109] x [1,91] are removed % -- %
keep = (data(:,1) >= 1) & (data(:, 1) <= 91) & (data(:,2) >= 1) & (data(:, 2) <= 109) & ...
    (data(:,3) >= 1) & (data(:, 3) <= 91);
data = data(keep, :);

ID = sort(data(:,4));

% -- % Convert voxel space coordinates to mm space 2x2x2 % -- %
xaxis = 1:2:182; data(:, 1) = xaxis(data(:, 1)); clear xaxis;
yaxis = 1:2:218; data(:, 2) = yaxis(data(:, 2)); clear yaxis;
zaxis = 1:2:182; data(:, 3) = zaxis(data(:, 3)); clear zaxis;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	 	 Load covariates matrix      % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cova = dlmread('covariates_encoded2.txt');                       % Loading file with covariates
r = size(cova, 2);									    % Number of covariates per study

%%%%%%%%%%%%%%%%%%%%%%%
% % Load study type % %
%%%%%%%%%%%%%%%%%%%%%%%
Y = dlmread('studytype.txt');                           % Study-type  
J = 1;                                                  % 

%%%%%%%%%%%%%%%%%%%%
% %	Rescale axes % % 
%%%%%%%%%%%%%%%%%%%%
% Axes are measured on a 4 x 4 x 4 mm grid, with 45 x 54 x 45 voxels 
% If a different grid is needed (to approx the HMC integral), the 
% following lines will need to be changed accordingly
Ix = 2:4:180;
Iy = 2:4:216;
Iz = 2:4:180;
% Volume of each voxel -- needed for HMC %
A = (Iy(2) - Iy(1))*(Ix(2) - Ix(1))*(Iz(2) - Iz(1));	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	Define bases: Gaussian kernels % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

knots = dlmread('knots.txt');

nu = 1/(2*(10)^2);

% -- % Observed matrix of basis functions % -- %
% Lines 80 - 134 define quantities used in HMC %
B = zeros(size(data,1),size(knots, 1));
for i = 1:size(data,1)
    obs_knot = repmat(data(i, 1:d), [size(knots, 1), 1]) - knots;
    for j = 1:size(knots, 1)
	B(i, j) = exp(-nu * norm(obs_knot(j,:))^2);
    end
end
B = [ones(size(data,1), 1) B];
nbasis = size(B, 2);	 % Number of basis functions

sum_B = [];              % Insert into HMC function the sum of bases
for h = 1:N
	if (size(B(ID == h, :), 1) * size(B(ID == h, :), 2)  == nbasis) 
        sum_B(h, :) = B(ID == h, :);
	else sum_B(h, :) = sum(B(ID == h, :));
    end
end

Bpred = dlmread('B.predcbma10.txt');

tBpred = Bpred';
clear B; clear knots;
V = size(Bpred, 1);         % Number of grid points

mkdir('enc2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	    Define global constants      % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nrun = 75000;                               % Number of MCMC iterations
burn = 25000;                               % Burn-in
thin = 25;                                  % Thinning MCMC samples
every = 100;                                % Number of previous samples consider to update the HMC stepsize
start = 250;                                 % Starting iteration to update the stepsize
sp = (nrun - burn)/thin;					% Number of posterior samples
epsilon = 1e-4;                             % Threshold limit (update of Lambda)
prop = 1.00;                                % Proportion of redundant elements within columns

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	 Define hyperparameter values    % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b0 = 2; b1 = 0.0001;                        % Gamma hyperprior params on a_1 and a_2
as = 1; bs = 0.3;							% Gamma hyperparameters for residual precision
df = 3;										% Gamma hyperparameters for t_{ij}
ad1 = 2.1; bd1 = 1;						    % Gamma hyperparameters for delta_1
ad2 = 3.1; bd2 = 1;						    % Gamma hyperparameters for delta_h, h >=2
adf = 1; bdf = 1;							% Gamma hyperparameters for ad1 and ad2 or df
epsilon_hmc = 0.0001;						% Starting valye for Leapfrog stepsize
L_hmc_def = 30;								% Leapfrog trajectory length

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	 		Initial values    		 % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = floor(log(nbasis)*2);					            % Number of latent factors to start with
sig = ones(nbasis,1)*0.5;                               % Residual variance (diagonal of Sigma^-1)
phiih = gamrnd(df/2, 2/df, [nbasis,k]);	                % Local shrinkage coefficients 
delta = [gamrnd(ad1, bd1); gamrnd(ad2, bd2, [k-1, 1])];	% Global shrinkage coefficients multipliers			
tauh = cumprod(delta);									% Global shrinkage coefficients (ex tau)
Plam = (phiih .* repmat(tauh',[nbasis, 1]));			% Precision of loadings rows (ex Ptht)
Lambda = zeros(nbasis, k);								% Matrix of factor loading
eta = normrnd(0, 1, k, N);                              % Matrix of latent factors
theta = mvnrnd((Lambda * eta(:, 1:N))', diag(1./sig));  % Matrix of basis function coefficients
iota = unifrnd(0, 1, r, k);                     		% Initialise matrix of covariates' coefficients 
Omega = zeros(r, k);            					    % Initialise Matrix of iota variances 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	DO mult probit extension: intial values & hyper.  % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTE: If you change mu_gamma or inv_sigma_gamma, modify 
% lines 416 and 417, 428, 429 accordingly (adaptation of k block)
latent = ones(N, J); 	                                % Latent variable indicators
mu_alpha = norminv(0.2); sigma_alpha = eye(J);			% Hyperparameters for normal prior on alpha
alpha = mvnrnd(mu_alpha, sigma_alpha);                  % Type-specific random intercept
post_var_alpha = diag(repmat(1/(N+1), [J,1]));          % Post cov matrix for update of alpha

gamma = ones(k, J);                                      % Gamma coefficients for DO probit model
mu_gamma = zeros(k, 1); Sigma_gamma = eye(k); 				    % Hyperparameters for normal prior on gamma
inv_sigma_gamma = inv(Sigma_gamma);								% Invert cov matrix for posterior computation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	Setting train & test sets  % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = sort([randperm(sum(Y==0),23),sum(Y==0) + randperm(sum(Y==1),15)]);       % Test set (20% of data points)
train = setdiff(1:N, test);                % Test set
ltrain = length(train);                    % Length of train set 
dlmwrite('enc2/train.txt', train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	     Define output files      % % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxk = 50;										% Max expected number of factors
Acc = zeros(N, nrun);						    % Acceptance probabilities

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %	  Start Gibbs sampling 	  % %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
for i = 1:nrun
    
    % -- Update Lambda -- %
    Lambda = zeros(nbasis,k);
    for j = 1:nbasis
        Vlam1 = diag(Plam(j,:)) + sig(j)*(eta*eta');  
        T = cholcov(Vlam1);
        [~,R] = qr(T);
        S = R\eye(size(R,1));
        Vlam = S*S';
        Elam = Vlam*sig(j)*eta*(theta(:,j));                                     
        Lambda(j,:) = (Elam + S*normrnd(zeros(k,1),1))';                                        
    end
    k = size(Lambda,2);
    
    % -- Update phi_{ih}'s -- %
    phiih = gamrnd(df/2 + 0.5, 1./(df/2 + bsxfun(@times,Lambda.^2, tauh')));
    
    % -- Update delta -- %
    ad = ad1 + nbasis*k/2;
    bd = bd1 + 0.5 * (1/delta(1)) * sum(tauh'.*(sum(phiih.*Lambda.^2)));
    delta(1) = gamrnd(ad,1/bd);

    tauh = cumprod(delta);
    for h = 2:k
        ad = ad2 + nbasis*(k-h+1)/2;
        temp1 = tauh'.*(sum(phiih.*Lambda.^2));
        bd = bd2 + 0.5 * (1/delta(h))*sum(temp1(h:k));
        delta(h) = gamrnd(ad,1/bd);
        tauh = cumprod(delta);
    end
    
	% -- Update precision parameters -- %
    Plam = bsxfun(@times,phiih,tauh');
    
%     % -- Update bd1 and bd2 -- %
%     % bd1 = gamrnd(ad1,1/delta(1));
%     % bd2 = gamrnd(ad2*(k-1),1/sum(delta(2:end)));
%     bd1 = gaminv(unifrnd(0,gamcdf(1,ad1,1/delta(1))),ad1,1/delta(1));
%     bd2 = gaminv(unifrnd(0,gamcdf(1,ad2*(k-1),1/sum(delta(2:end)))),ad2*(k-1),1/sum(delta(2:end)));
% 
%     % -- Update ad1 and ad2 -- %
%     % Sample a candidate value for log(ad1) from truncated normal
%     % ldfs = norminv(unifrnd(normcdf(0,log(ad1),0.2),1),log(ad1),0.2);
%     ldfs = normrnd(log(ad1),0.2);
%     ratio = exp(log(gampdf(delta(1),exp(ldfs),1/bd1)) - log(gampdf(delta(1),ad1,1/bd1)) + ...
%         log(gampdf(exp(ldfs),adf,1/bdf)) - log(gampdf(ad1,adf,1/bdf)));
%     if unifrnd(0,1)< ratio && ldfs > 0
%        ad1 = exp(ldfs);
%     end
%         
%     % Sample a candidate value for log(ad2) from truncated normal
%     % ldfs = norminv(unifrnd(normcdf(0,log(ad2),0.2),1),log(ad2),0.2);
%     ldfs = normrnd(log(ad2),0.2);
%     ratio = exp(log(gampdf(delta(2:k),exp(ldfs),1/bd2))'*ones(k-1,1) - log(gampdf(delta(2:k),ad2,1/bd2))'*ones(k-1,1) + ...
%         log(gampdf(exp(ldfs),adf,1/bdf)) - log(gampdf(ad2,adf,1/bdf)));
%     if unifrnd(0,1)< ratio && ldfs > 0
%        ad2 = exp(ldfs);
%     end
%     
%     % -- Update df -- %
%     % Sample a candidate value for log degrees of freedom
%     % ldfs = norminv(unifrnd(normcdf(0,log(df),0.2),1),log(df),0.2);   
%     ldfs = normrnd(log(df),0.2); px = size(phiih(:), 1);
%     ratio = exp(log(gampdf(phiih(:),exp(ldfs)/2,2/exp(ldfs)))'*ones(px,1) - log(gampdf(phiih(:),df/2,2/df))'*ones(px,1) + ...
%         log(gampdf(exp(ldfs),adf,1/bdf)) - log(gampdf(df,adf,1/bdf)));
%     if unifrnd(0,1)<ratio
%        df = exp(ldfs);
%     end

    % -- Update Sigma precisions -- %
    thetatil = theta' - Lambda*eta;
    sig = gamrnd(as + N/2, 1./(bs + 0.5*sum(thetatil.^2,2)));
    
    % -- Update linear model on latent factors -- %
	for l = 1:k
      Omega(:, l) =  gamrnd(1, 1./(0.5 * (1 + iota(:,l).^2)));
      Veta1 = cova' * cova + diag(Omega(:,l));     
      T = cholcov(Veta1); [~, R] = qr(T); S = R\eye(size(R,1)); Vlam = S * S';
      Meta = Vlam * (eta(l,:) * cova)';
      iota(:,l) = Meta + S * normrnd(0,1, [r, 1]); 
     end
    
    % -- Update of eta (probit model extension with covariates) -- %
    Lmsg = Lambda' * diag(sig);
	Veta1 = Lmsg * Lambda + eye(k) + gamma * gamma';
	T = cholcov(Veta1);
    [~,R] = qr(T);
    S = R\eye(size(R,1));
    Vlam = S*S';        
    Meta = Vlam * (Lmsg * theta' + gamma * (latent - repmat(alpha, [N, 1]))' + (cova * iota)');                                   
    eta = Meta + S * normrnd(0,1,[k,N]); 

    % -- Update alpha intercept -- %
    post_mean_alpha = sigma_alpha * mu_alpha + sum(latent - eta'*gamma)';
	alpha = mvnrnd(post_var_alpha * post_mean_alpha, post_var_alpha);
    
    % -- Update gamma coefficients -- %
	Veta1 = inv_sigma_gamma + eta * eta';
    T = cholcov(Veta1); [~, R] = qr(T); S = R\eye(size(R,1)); Vlam = S * S';
    Meta = Vlam * (repmat(inv_sigma_gamma * mu_gamma, [1, J]) + eta * (latent - repmat(alpha, [N, 1]))); 
	gamma = Meta + S * normrnd(zeros(k,J),1);

    % -- Update latent indicators -- %
    mean_latent = repmat(alpha, [N, 1]) + eta' * gamma;
 	latent = zeros([N, J]);
    
    % % % % % % % % % % % % % % % % % % % % % % % % 
    % Take care of studies in the train set first %
    % % % % % % % % % % % % % % % % % % % % % % % %
    for j = 1:ltrain
         quale = train(j);
         ind = Y(quale);
         if ind == 0
            latent(quale, 1) = randraw('normaltrunc', [-Inf, 0, mean_latent(quale, 1), 1], 1);
         else
            latent(quale, 1) = randraw('normaltrunc', [0, Inf, mean_latent(quale, 1), 1], 1);
         end
    end
    
    % % % % % % % % % % % % % % % % % % % % % 
    % Test set via predictive probabilities % --> DO mult probit model
    % % % % % % % % % % % % % % % % % % % % % 
    
    pred_prob = 1 - normcdf(mean_latent(test,:), 0, 1);    % Probability of being of type 0
    u = unifrnd(0, 1, [length(test), 1]);
    pred_cat = bsxfun(@gt, u, pred_prob)*1;  
    
    for j = 1:length(test)
         quale = test(j);
         ind = pred_cat(j);
         if ind == 0
            latent(quale, 1) = randraw('normaltrunc', [-Inf, 0, mean_latent(quale, 1), 1], 1);
         else
            latent(quale, 1) = randraw('normaltrunc', [0, Inf, mean_latent(quale, 1), 1], 1);
         end
    end

    if mod(i,thin) == 0 
       dlmwrite('enc2/pred_prob.txt', pred_prob(:)', 'delimiter', ' ', '-append');
       dlmwrite('enc2/pred_cat.txt', pred_cat', 'delimiter', ' ', '-append');
    end

    % -- Update of theta -- #
	L_hmc = poissrnd(L_hmc_def);
    dlmwrite('enc2/HMC_nsteps.txt', L_hmc, 'delimiter', ' ', '-append');

    [theta_hmc, acc, pot_energy] = HMC(epsilon_hmc, L_hmc, theta, A, Bpred, tBpred, Lambda, eta, sig, sum_B);

    if sum(isnan(acc)) ~= 0 
       acc(isnan(acc), 1) = 0;
    end
    
    theta = theta_hmc;
	Acc(:, i) = acc;
	
 	dlmwrite('enc2/Acc_while.txt', mean(Acc(:, i)), '-append');
    
    if mod(i,every) == 0 && i <= burn
       dlmwrite('enc2/thetacbma_pre.txt', (theta(:))', 'delimiter', ' ', '-append');
    end
 	
    if mod(i,thin) == 0 && i > burn
       dlmwrite('enc2/thetacbma_post.txt', (theta(:))', 'delimiter', ' ', '-append');
    end
 	
	if mod(i,thin) == 0 && i <= burn && i >= start
		avr = mean(mean(Acc(:, ((i - start) + 1) : i)));
		epsilon_hmc = (avr >= 0.65) * 1.05 * epsilon_hmc + (avr < 0.65) * 0.95 * epsilon_hmc; 		
		dlmwrite('enc2/Eps.txt', epsilon_hmc, 'delimiter', ' ', '-append');
    end
        
    % -- Adapt number of latent factors -- %
	prob = 1/exp(b0 + b1*i);				      % Probability of adapting
	uu = rand;
	lind = sum(abs(Lambda) < epsilon)/nbasis;     % Proportion of elements in each column less than eps in magnitude
	vec = lind >= prop; num = sum(vec);           % number of redundant columns
    
     if uu < prob
            if  i > 20 && num == 0 && all(lind < 0.995)
                k = k + 1;
                Lambda(:,k) = zeros(nbasis,1);
                eta(k, :) = normrnd(0,1,[1, N]);           
                Omega(:, k) = gamrnd(.5, 2, [r,1]);
                iota(:, k) = mvnrnd(zeros(1,r), diag(1./(Omega(:,k))))';
                phiih(:,k) = gamrnd(df/2, 2/df,[nbasis,1]);
                delta(k) = gamrnd(ad2, 1/bd2);
                tauh = cumprod(delta);
			    Plam = bsxfun(@times, phiih, tauh'); 
                gamma(k,:) = normrnd(0,1,[J, 1])';
                mu_gamma = zeros(k, 1);
			    inv_sigma_gamma = eye(k); 				% Covariance for normal prior on gamma
            elseif num > 0
                nonred = setdiff(1:k, find(vec));
                k = max(k - num,1);
                Lambda = Lambda(:,nonred);
                eta = eta(nonred,:);
                phiih = phiih(:,nonred);
                delta = delta(nonred);
                tauh = cumprod(delta);
                Plam = bsxfun(@times, phiih, tauh');
                gamma = gamma(nonred,:);
                mu_gamma = zeros(k, 1);
                inv_sigma_gamma = eye(k); 				% Covariance for normal prior on gamma
                iota = iota(:, nonred);
                Omega = Omega(:, nonred);
           end
     end
        
   % -- Save sampled values (after thinning) -- %
    if mod(i,every) == 0 && i <= burn
        dlmwrite('enc2/sigma.txt', sig', 'delimiter', ' ', '-append');
        dlmwrite('enc2/alpha.txt', alpha, 'delimiter', ' ', '-append');
        dlmwrite('enc2/Factor.txt', k, 'delimiter', ' ', '-append');
        
        Etaout_PreBIN = zeros(N*maxk, 1);  
        teta = eta'; Etaout_PreBIN(1:(N*k), 1) = teta(:); clear teta;
        dlmwrite('enc2/Eta_PreBIN.txt', Etaout_PreBIN', 'delimiter', ' ', '-append'); clear Etaout_PreBIN;
 	    
        Gammaout_PreBIN = zeros(J*maxk, 1); tgamma = gamma'; Gammaout_PreBIN(1:(J*k), 1) = tgamma(:); clear tgamma;
        dlmwrite('enc2/Gamma_PreBIN.txt', Gammaout_PreBIN', 'delimiter', ' ', '-append'); clear Gammaout_PreBIN;
 	    
        Iotaout_PreBIN = zeros(r*maxk, 1); Iotaout_PreBIN(1:(r*k), 1) = iota(:);
        dlmwrite('enc2/Iota_PreBIN.txt', Iotaout_PreBIN', 'delimiter', ' ', '-append'); clear Iotaout_PreBIN;
        
        Omegaout_PreBIN = zeros(r*maxk, 1); Omegaout_PreBIN(1:(r*k), 1) = Omega(:);
 	    dlmwrite('enc2/Omega_PreBIN.txt', Omegaout_PreBIN', 'delimiter', ' ', '-append'); clear Omegaout_PreBIN;
        % dlmwrite('enc2/mgpshyper.txt', [ad1, bd1, ad2, bd2, df], 'delimiter', ' ', '-append');
        phiihout = zeros(nbasis * maxk, 1); phiihout(1:(nbasis * k), 1) = phiih(:);
        dlmwrite('enc2/Phiih.txt', phiihout', 'delimiter', ' ', '-append'); clear phiihout;
        deltaout = zeros(maxk, 1); deltaout(1:k, 1) = delta;
        dlmwrite('enc2/Delta.txt', deltaout', 'delimiter', ' ', '-append'); clear deltaout;
    end
   if mod(i,thin) == 0 && i > burn
 	   Lambdaout = zeros(nbasis*maxk, 1); Lambdaout(1:(nbasis*k), 1) = Lambda(:)';
       dlmwrite('enc2/Lambda.txt', Lambdaout', 'delimiter', ' ', '-append'); clear Lambdaout;
 	   
       Etaout = zeros(N*maxk, 1); teta = eta'; Etaout(1:(N*k), 1) = eta(:); clear teta;
       dlmwrite('enc2/Eta_PBIN.txt', Etaout', 'delimiter', ' ', '-append'); clear Etaout;
 	   
       dlmwrite('enc2/HMC_Energy.txt', pot_energy', 'delimiter', ' ', '-append'); 
       
       Gammaout = zeros(J*maxk, 1); tgamma = gamma'; Gammaout(1:(J*k), 1) = tgamma(:); clear tgamma;
 	   dlmwrite('enc2/Gamma_PBIN.txt', Gammaout', 'delimiter', ' ', '-append'); clear Gammaout;
       
       Iotaout = zeros(r*maxk, 1); Iotaout(1:(r*k), 1) = iota(:);
       dlmwrite('enc2/Iota_PBIN.txt', Iotaout', 'delimiter', ' ', '-append'); clear Iotaout;
 	   
       Omegaout_PBIN = zeros(r*maxk, 1); Omegaout_PBIN(1:(r*k), 1) = Omega(:);
       dlmwrite('enc2/Omega.txt', Omegaout_PBIN', 'delimiter', ' '); clear Omegaout_PBIN;
 		
       dlmwrite('enc2/sigma.txt', sig', 'delimiter', ' ', '-append');
       dlmwrite('enc2/alpha.txt', alpha, 'delimiter', ' ', '-append');
       dlmwrite('enc2/Factor.txt', k, 'delimiter', ' ', '-append');
       dlmwrite('enc2/latent.txt', latent(:)', 'delimiter', ' ', '-append');
       % dlmwrite('enc2/mgpshyper.txt', [ad1, bd1, ad2, bd2, df], 'delimiter', ' ', '-append');
       phiihout = zeros(nbasis * maxk, 1); phiihout(1:(nbasis * k), 1) = phiih(:);
       dlmwrite('enc2/Phiih.txt', phiihout', 'delimiter', ' ', '-append'); clear phiihout;
       deltaout = zeros(maxk, 1); deltaout(1:k, 1) = delta;
       dlmwrite('enc2/Delta.txt', deltaout', 'delimiter', ' ', '-append'); clear deltaout;
  end
	[i, k]
end
toc;
dlmwrite('enc2/HMC_Acc.txt', Acc', 'delimiter', ' ');

