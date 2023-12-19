function [V,D,bnd,j,work] = laneig(A,nin,k,sigma,options)
%clc;clear all;close all;
%n = 10;
%A = randn(n);
%A = A+A';
%[vector,value] = laneig(A,1,'AL');
%value

warning off;
%LANEIG  Compute a few eigenvalues and eigenvectors.
%   LANEIG solves the eigenvalue problem A*v=lambda*v, when A is 
%   real and symmetric using the Lanczos algorithm with partial 
%   reorthogonalization (PRO). 
%
%   [V,D] = LANEIG(A) 
%   [V,D] = LANEIG('Afun',N) 
%
%   The first input argument is either a real symmetric matrix, or a 
%   string containing the name of an M-file which applies a linear 
%   operator to the columns of a given matrix.  In the latter case,
%   the second input argument must be N, the order of the problem.
%
%   The full calling sequence is
%
%   [V,D,ERR] = LANEIG(A,K,SIGMA,OPTIONS)
%   [V,D,ERR] = LANEIG('Afun',N,K,SIGMA,OPTIONS)
%
%   On exit ERR contains the computed error bounds.  K is the number of
%   eigenvalues desired and SIGMA is numerical shift or a two letter string
%   which specifies which part of the spectrum should be computed:
%
%   SIGMA            Specified eigenvalues
%
%   'AL'            Algebraically Largest 
%   'AS'            Algebraically Smallest
%   'LM'            Largest Magnitude   (default)
%   'SM'            Smallest Magnitude  (does not work when A is an m-file)
%   'BE'            Both Ends.  Computes k/2 eigenvalues
%                   from each end of the spectrum (one more
%                   from the high end if k is odd.) 
%
%   The OPTIONS structure specifies certain parameters in the algorithm.
%
%    Field name      Parameter                              Default
%   
%    OPTIONS.tol     Convergence tolerance                  16*eps
%    OPTIONS.lanmax  Dimension of the Lanczos basis.
%    OPTIONS.v0      Starting vector for the Lanczos        rand(n,1)-0.5
%                    iteration.
%    OPTIONS.delta   Level of orthogonality among the       sqrt(eps/K)
%                    Lanczos vectors.
%    OPTIONS.eta     Level of orthogonality after           10*eps^(3/4)
%                    reorthogonalization. 
%    OPTIONS.cgs     reorthogonalization method used        0
%                    '0' : iterated modified Gram-Schmidt 
%                    '1' : iterated classical Gram-Schmidt
%    OPTIONS.elr     If equal to 1 then extended local      1
%                    reorthogonalization is enforced. 
%
%   See also LANPRO, EIGS, EIG.

% References: 
% R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
%
% B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
% Prentice-Hall, Englewood Cliffs, NJ, 1980.
%
% H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
% Math. Comp. 42 (1984), no. 165, 115--142.

% Rasmus Munk Larsen, DAIMI, 1998


%%%%%%%%%%%%%%%%%%%%% Parse and check input arguments. %%%%%%%%%%%%%%%%%%%%%%

if ~isstr(A)
  if nargin<1
    error('Not enough input arguments.');
  end
  [m n] = size(A);
  Aisfunc = 0;
  if m~=n | ~isequal(A,A') | ~isreal(A)
    error('A must be real symmetric')
  end  
  if nargin < 4 | isempty(sigma)
    options = [];
  else  
    options = sigma; 
  end
  if nargin < 3 | isempty(k), sigma = 'LM'; else, sigma = k; end
  if nargin < 2 | isempty(nin), k = min(n,5); else, k = nin; end
else
  if nargin<2
    error('Not enough input arguments.');
  end
  Aisfunc = 1;
  n = nin;
  if nargin < 5 | isempty(options)
    options.tol = 16*eps;
    options.lanmax = n;
    options.v0 = rand(n,1)-0.5;
  end
  if nargin < 4 | isempty(sigma), sigma = 'LM'; end
  if nargin < 3 | isempty(k), k = min(n,5);  end
end

if ~isnumeric(k) | real(abs(fix(k)))~=k | ~isnumeric(n) | real(abs(fix(n)))~=n
  error('Input arguments N and K must be positive integers.')
end

% Quick return for n<2  or k<1
if n < 1 | k<1
  if nargout < 2
    V = zeros(k,1);
  else
    V = eye(n,k);
    D = zeros(k,k);
    bnd =zeros(k,1);
  end
  return
end
if n == 1 
  if ~Aisfunc
    D = A;
    V = 1;
    bnd = 0;
  else
    D = feval(A,1);
    V = 1;
    dnb = 0;
  end
  if nargout<2
    V=D;
  end
  return
end

% A is the matrix of all zeros (not detectable if A is a string)
if ~Aisfunc 
  if nnz(A)==0
    if nargout < 2
      V = zeros(k,1);
    else
      V = eye(n,k);
      D = zeros(k,k);
      bnd =zeros(k,1);
    end
    return
  end
end

lanmax = n;
tol = 16*eps;
r = rand(n,1)-0.5;
part = sigma;
% Parse options struct
if ~isempty(options) & isstruct(options)
  c = fieldnames(options);
  for i=1:length(c)
    if strmatch(c(i),'v0'), r = getfield(options,'v0'); r=r(:); end
    if strmatch(c(i),'tol'), tol = getfield(options,'tol'); end
    if strmatch(c(i),'lanmax'), lanmax = getfield(options,'lanmax'); end
  end
end

% Protect against absurd arguments.
tol = max(tol,eps);
lanmax = min(lanmax,n);
if size(r,1)~=n
  error('v0 must be a vector of length n')
end

lanmax = min(lanmax,n);
if k>lanmax
  error('K must satisfy  K <= LANMAX <= N.');
end
ksave = k;

if strcmp(sigma,'SM') & ~isstr(A)
  sigma = 0;
end


% Prepare for shift-and-invert if sigma is numeric.
if  isnumeric(sigma)
  part = 'LM';
  if isstr(A) 
    error('Shift-and-invert works only when the matrix A is given explicitly.');
  else
    pmmd = symmmd(A);
    A = A(pmmd,pmmd);
    [S.L,S.U] = lu(A - sigma*speye(n));
    condU = condest(S.U);
    dsigma = n * full(max(max(abs(A)))) * eps;
    if sigma < 0
      sgnsig = -1;
    else
      sgnsig = 1;
    end
    sigitr = 1;
    while condU > 1/eps & ((dsigma <= 1 & sigitr <= 10) | ~isfinite(condU))
      disps1 = sprintf(['sigma = %10e is near an exact eigenvalue of A,\n' ...
			'so we cannot use the LU factorization of (A-sigma*I): ' ...
			' condest(U) = %10e.\n'],sigma,condU);
      if abs(sigma) < 1
	sigma = sigma + sgnsig * dsigma;
	disps2 = sprintf('We are trying sigma + %10e = %10e instead.\n', ...
			 sgnsig*dsigma,sigma);
      else
	sigma = sigma * (1 + dsigma);
	disps2 = sprintf('We are trying sigma * (1 + %10e) = %10e instead.\n', ...
			 dsigma,sigma);
      end
      %     if nargout < 3 & dispn ~= 0             
      disp([disps1 disps2])
      %     end   
      [S.L,S.U] = lu(A - sigma*speye(n));
      condU = condest(S.U);
      dsigma = 10 * dsigma;
      sigitr = sigitr + 1;
    end
  end
  A = S;
end


neig = 0; nrestart=-1;
if ~strcmp(part,'BE') 
  j = min(2*k+2,lanmax);
else
  j = min(k+1,lanmax);
end


%%%%%%%%%%%%%%%%%%%%% Here begins the computation  %%%%%%%%%%%%%%%%%%%%%%

V = []; T = []; anorm = []; work = zeros(1,2); rnorm=-1;




while neig < k 
  %%%%%%%%%%%%%%%%%%%%% Compute Lanczos tridiagonalization %%%%%%%%%%%%%%%%%
  j = min(lanmax,j+1-mod(j,2));
  % "Trick" to avoid unwanted zero eigenvalues when laneig is used for
  % SVD calculations. (Nothing to if lanmax is odd, though.)
  
  if  ~isstr(A)
    [V,T,r,anorm,ierr,w] = lanpro(A,j,r,options,V,T,anorm);
  else
    [V,T,r,anorm,ierr,w] = lanpro(A,n,j,r,options,V,T,anorm);
  end
  work= work + w;

  if ierr<0 % Invariant subspace of dimension -ierr found. 
    j = -ierr;
  end

  %%%%%%%%%%%%%%%%%% Compute eigenvalues and error bounds %%%%%%%%%%%%%%%%%%
  % Analyze T
  [D,top,bot,err] = tqlb([full(diag(T))],full([0;diag(T,1)]));
  %  if err>0
  %    printf(['TQLB failed. Eigenvalue no. %i did not converge in 30', ...
  %	  ' iterations'],err);
  %  end
  %  full(T)
  %  [P,D] = eig(full(T));
  %  D = diag(D);
  %  bot = P(end,:)';
  %  [P(1,:)' P(end,:)']
  [D,I] = sort(D);
  bot = bot(I);
  
  % Set simple error bounds
  rnorm = norm(r);
  bnd = rnorm*abs(bot);
  
  % Use Largest Ritz value to estimate ||A||_2. This might save some
  % reorth. in case of restart.
  anorm = max(abs(D));
  
  % Estimate gap structure and refine error bounds
  bnd = refinebounds(D,bnd,n*eps*anorm);

  %%%%%%%%%%%%%%%%%%% Check convergence criterion %%%%%%%%%%%%%%%%%%%%
  % Reorder eigenvalues according to SIGMA
  switch part
   case 'AS'
    IPART = 1:j;
   case 'AL' 
    IPART = j:-1:1;
   case 'LM'
    [dummy,IPART] = sort(-abs(D));
   case 'BE'
    if j<k
      IPART=1:j;
    else
      mid = floor(k/2);
      par = rem(k,1);
      IPART = [1:mid,(j-mid-par):j]';
    end    
   otherwise
    error(['Illegal value for SIGMA: ',part]);
  end
  D = D(IPART);  bnd = bnd(IPART);
  if isnumeric(sigma)
    D = sigma + 1./D;
  end
  
  % Check if enough have converged.
  neig = 0;
  for i=1:min(j,k)
    if bnd(i) <= tol*abs(D(i))
      neig = neig + 1;
    end
  end
  
  %%%%%%%%%%% Check whether to stop or to extend the Krylov basis? %%%%%%%%%%
  if ierr<0 % Invariant subspace found
    if j<k
      warning(['Invariant subspace of dimension ',num2str(j-1),' found.'])
    end
    break;
  end
  if j>=lanmax % Maximal dimension of Krylov subspace reached => Bail out!
    if neig<ksave
      warning(['Maximum dimension of Krylov subspace exceeded prior',...
	       ' to convergence.']);
    end
    break;
  end
  
  % Increase dimension of Krylov subspace and try again.
  if neig>0
    %    j = j + ceil(min(20,max(2,((j-1)*(k-neig+1))/(2*(neig+1)))));
    j = j + min(100,max(2,0.5*(k-neig)*j/(neig+1)));
  elseif neig<k
    %    j = j + ceil(min(20,max(8,(k-neig)/2)));
    j = max(1.5*j,j+10);
  end
  j = min(j+1,lanmax);
  nrestart = nrestart + 1;
end



%%%%%%%%%%%%%%%% Lanczos converged (or failed). Prepare output %%%%%%%%%%%%%%%
k = min(ksave,j);

if nargout>1
  j = size(T,1);
  [Q,D] = eig(full(T)); D = diag(D);
  [D,I] = sort(D);
  % Compute and normalize Ritz vectors (overwrite V to save memory).
  V = V*Q(:,I(IPART(1:k)));
  for i=1:k
    nq = norm(V(:,i));
    if isfinite(nq) & nq~=0 & nq~=1
      V(:,i) = V(:,i)/nq;
    end
  end
  [D,I] = sort(D);
  D = D(IPART(1:k));
  if isnumeric(sigma)
    D = sigma + 1./D;
    V(pmmd,:) = V;
  end
end

% Pick out desired part of the spectrum
if length(D)~=k
  D = D(1:k);
  bnd = bnd(1:k);
end

if nargout<2
  V = D;
else
  D = diag(D);
end


function [Q_k,T_k,r,anorm,ierr,work] = lanpro(A,nin,kmax,r,options,...
    Q_k,T_k,anorm)
 
%LANPRO   Lanczos tridiagonalization with partial reorthogonalization
%   LANPRO computes the Lanczos tridiagonalization of a real symmetric 
%   matrix using the symmetric Lanczos algorithm with partial 
%   reorthogonalization. 
%
%   [Q_K,T_K,R,ANORM,IERR,WORK] = LANPRO(A,K,R0,OPTIONS,Q_old,T_old)
%   [Q_K,T_K,R,ANORM,IERR,WORK] = LANPRO('Afun',N,K,R0,OPTIONS,Q_old,T_old)
%
%   Computes K steps of the Lanczos algorithm with starting vector R0, 
%   and returns the K x K tridiagonal T_K, the N x K matrix Q_K 
%   with semiorthonormal columns and the residual vector R such that 
%
%        A*Q_K = Q_K*T_K + R .
%
%   Partial reorthogonalization is used to keep the columns of Q_K 
%   semiorthogonal:
%        MAX(DIAG((eye(k) - Q_K'*Q_K))) <= OPTIONS.delta.
%
%
%   The first input argument is either a real symmetric matrix, a struct with
%   components A.L and A.U or a string containing the name of an M-file which 
%   applies a linear operator to the columns of a given matrix.  In the latter
%   case, the second input argument must be N, the order of the problem.
%
%   If A is a struct with components A.L and A.U, such that 
%   L*U = (A - sigma*I), a shift-and-invert Lanczos iteration is performed
%
%   The OPTIONS structure is used to control the reorthogonalization:
%     OPTIONS.delta:  Desired level of orthogonality 
%                     (default = sqrt(eps/K)).
%     OPTIONS.eta  :  Level of orthogonality after reorthogonalization 
%                     (default = eps^(3/4)/sqrt(K)).
%     OPTIONS.cgs  :  Flag for switching between different reorthogonalization
%                     algorithms:
%                      0 = iterated modified Gram-Schmidt  (default)
%                      1 = iterated classical Gram-Schmidt 
%     OPTIONS.elr  :  If OPTIONS.elr = 1 (default) then extended local
%                     reorthogonalization is enforced.
%     OPTIONS.Y    :  The lanczos vectors are reorthogonalized against
%                     the columns of the matrix OPTIONS.Y.
%
%   If both R0, Q_old and T_old are provided, they must contain 
%   a partial Lanczos tridiagonalization of A on the form
%
%        A Q_old = Q_old T_old + R0 .  
%
%   In this case the factorization is extended to dimension K x K by
%   continuing the Lanczos algorithm with R0 as starting vector.
%
%   On exit ANORM contains an approximation to ||A||_2. 
%     IERR = 0  :  K steps were performed succesfully.
%     IERR > 0  :  K steps were performed succesfully, but the algorithm
%                  switched to full reorthogonalization after IERR steps.
%     IERR < 0  :  Iteration was terminated after -IERR steps because an
%                  invariant subspace was found, and 3 deflation attempts 
%                  were unsuccessful.
%   On exit WORK(1) contains the number of reorthogonalizations performed, and
%   WORK(2) contains the number of inner products performed in the
%   reorthogonalizations.
%
%   See also LANEIG, REORTH, COMPUTE_INT

% References: 
% R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
%
% G. H. Golub & C. F. Van Loan, "Matrix Computations",
% 3. Ed., Johns Hopkins, 1996.  Chapter 9.
%
% B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
% Prentice-Hall, Englewood Cliffs, NJ, 1980.
%
% H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
% Math. Comp. 42 (1984), no. 165, 115--142.

% Rasmus Munk Larsen, DAIMI, 1998


% Check input arguments.
if nargin<1, error('Not enough input arguments.');  end
if isnumeric(A) | isstruct(A)
  if isnumeric(A)
    [m n] = size(A);
    if m~=n | ~isequal(A,A') | ~isreal(A)
      error('A must be real symmetric')
    end  
  elseif isstruct(A)
    [m n] = size(A.L);
  end
    
  if nargin<7 | isempty(T_k), 
    anorm = []; est_anorm=1; 
  else
    anorm = T_k; est_anorm=0; 
  end
  if nargin<6,  Q_k=[]; T_k=[]; else,  T_k = Q_k; Q_k = options; end
  if nargin<4 | isempty(r),  options = []; else,  options = r;  end
  if nargin<3 | isempty(kmax),  
    r = rand(n,1)-0.5;
  else
    r = kmax;
  end
  if nargin<2 | isempty(nin);  kmax = max(10,n/10); else,  kmax = nin;  end   
else
  if nargin<2
    error('Not enough input arguments.');
  end
  % Check input functions and parse to create an internal object
  % if an explicit expression is given.
  [A, msg] = fcnchk(A);
  if ~isempty(msg)
    error(msg);
  end  
  n = nin;
  if nargin<8 | isempty(anorm), anorm = []; est_anorm=1; else est_anorm=0; end
  if nargin<7,  Q_k=[]; T_k=[]; end
  if nargin<5 | isempty(options),  options = [];          end
  if nargin<4 | isempty(r),  r = rand(n,1)-0.5;   end
  if nargin<3 | isempty(kmax);  kmax = max(10,n/10); end
end
 
% Set options.  
delta = sqrt(eps/kmax); % Desired level of orthogonality.
eta = eps^(3/4)/sqrt(kmax);     % Level of orth. after reorthogonalization.
cgs = 0;                % Flag for switching between iterated CGS and MGS.
elr = 1;                % Flag for switching extended local 
                        % reorthogonalization on and off.
deflate = 0;              % Flag for deflation against OPTIONS.Y
			
% Parse options struct
if ~isempty(options) & isstruct(options)
  c = fieldnames(options);
  for i=1:length(c)
    if strmatch(c(i),'delta'), delta = getfield(options,'delta');  end
    if strmatch(c(i),'eta'), eta = getfield(options,'eta'); end
    if strmatch(c(i),'cgs'), cgs = getfield(options,'cgs'); end
    if strmatch(c(i),'elr'), elr = getfield(options,'elr'); end
    if strmatch(c(i),'Y'), deflate = ~isempty(options.Y);  end
  end
end

np = 0;  nr = 0; ierr=0;

% Rule-of-thumb estimate on the size of round-off terms:
eps1 = sqrt(n)*eps/2; % Notice that {\bf u} == eps/2.
gamma = 1/sqrt(2);

% Prepare Lanczos iteration
if isempty(Q_k) % New Lanczos tridiagonalization.
  % Allocate space 
  alpha = zeros(kmax+1,1);  beta = zeros(kmax+1,1);
  Q_k = zeros(n,kmax);
  q = zeros(n,1); beta(1)=norm(r);
  omega = zeros(kmax,1); omega_max = omega;  omega_old = omega;
  omega(1) = 0;   force_reorth= 0;  
  j0 = 1;
else            % Extending existing Lanczos tridiagonalization.
  j = size(Q_k,2); % Size of existing factorization
  % Allocate space
  Q_k = [Q_k zeros(n,kmax-j)]; 
  alpha = zeros(kmax+1,1);  beta = zeros(kmax+1,1);
  alpha(1:j) = diag(T_k);  
  if j>1
    beta(2:j) = diag(T_k,-1);
  end
  q = Q_k(:,j);
  % Reorthogonalize r.
  beta(j+1) = norm(r);
  if j<kmax & beta(j+1)*delta < anorm*eps1,
    fro = 1;
  end
  if isfinite(delta)
    int = 1:j;
    [r,beta(j+1),rr] = reorth(Q_k,r,beta(j+1),int,gamma,cgs);
    np = rr*j;    nr = 1;   force_reorth = 1;  
  else
     force_reorth = 0;  
  end
  % Compute Gerscgorin bound on ||T_k||_2 as SQRT(||T_k'*T_k||_1)
  if est_anorm
    anorm = sqrt(norm(T_k'*T_k,1));
  end
  omega = eps1*ones(kmax,1); omega_max = omega;  omega_old = omega;
  j0 = j+1;
end

if delta==0
  fro = 1; % The user has requested full reorthogonalization.
else
  fro = 0;
end

for j=j0:kmax,  
  % Lanczos Step:
  q_old = q;
  if beta(j)==0
    q = r;
  else
    q = r / beta(j);
  end
  Q_k(:,j) = q;
  if isnumeric(A)
    u = A*q;
  elseif isstruct(A)
    u = A.U \ ( A.L \ q);
  else
    u = feval(A,q);
  end
  r = u - beta(j)*q_old;
  alpha(j) = q'*r;
  r = r - alpha(j)*q;
  

  % Extended local reorthogonalization:
  beta(j+1) = sqrt(r'*r); % Quick and dirty estimate.
  if beta(j+1)<gamma*beta(j) & elr 
    if  j==1
      t1=0;
      for i=1:2
	t = q'*r;    
	r = r-q*t;
	t1 = t1+t;
      end
      alpha(j) = alpha(j) + t1;
    elseif j>1
      t1 = q_old'*r;
      t2 = q'*r;
      r = r  - (q_old*t1 + q*t2); % Add small terms together first to
      if beta(j)~=0               % reduce risk of cancellation.
	beta(j) = beta(j) + t1;
      end
      alpha(j) = alpha(j) + t2;
    end        
    beta(j+1) = sqrt(r'*r); % Quick and dirty estimate.
  end

  % Update Gersgorin estimate of ||T_k|| if required
%  if est_anorm & beta(j+1)~=0
%    T_k = spdiags([[beta(2:j);0] alpha(1:j) beta(1:j)],-1:1,j,j);
%    anorm = sqrt(norm(T_k'*T_k,1))
%  end
  if  est_anorm & beta(j+1)~=0
    anorm = update_gbound(anorm,alpha,beta,j);
  end

  % Update omega-recurrence
  if j>1 & ~fro & beta(j+1)~=0
    [omega,omega_old] = update_omega(omega,omega_old,j,alpha,beta,...
	eps1,anorm);
    omega_max(j) = max(abs(omega));
  end

  % Reorthogonalize if required
  if j>1 & (fro  | force_reorth | omega_max(j)>delta) & beta(j+1)~=0
    if fro
      int = 1:j;
    else
      if force_reorth == 0
	force_reorth= 1; % Do forced reorth to avoid spill-over from q_{j-1}.
	int = compute_int(omega,j,delta,eta,0,0,0);
      else
	force_reorth= 0; 
      end
    end
    [r,beta(j+1),rr] = reorth(Q_k,r,beta(j+1),int,gamma,cgs);
    omega(int) = eps1;
    np = np + rr*length(int(:));    nr = nr + 1;
  else
    beta(j+1) = norm(r); % compute norm accurately.
  end

  if deflate    
    [r,beta(j+1),rr] = reorth(options.Y,r,beta(j+1),1:size(options.Y,2), ...
			      gamma,cgs);
  end
  
  if  j<kmax & beta(j+1) < n*anorm*eps  , 
    % If beta is "small" we deflate by setting the off-diagonals of T_k
    % to 0 and attempt to restart with a basis for a new 
    % invariant subspace by replacing r with a random starting vector:
    beta(j+1) = 0;
    bailout = 1;
    for attempt=1:3    
      r = rand(n,1)-0.5;  
      if isnumeric(A)
	r = A*r;
      elseif isstruct(A)
	r = A.U \ ( A.L \ r);
      else
	r = feval(A,r);
      end      
      nrm=sqrt(r'*r); % not necessary to compute the norm accurately here.
      int = 1:j;
      [r,nrmnew,rr] = reorth(Q_k,r,nrm,int,gamma,cgs);
      omega(int) = eps1;
      np = np + rr*length(int(:));    nr = nr + 1;
      if nrmnew > 0
	% A vector numerically orthogonal to span(Q_k(:,1:j)) was found. 
	% Continue iteration.
	bailout=0;
	break;
      end
    end
    if bailout
      ierr = -j;
      break;
    else
      r=r/nrmnew; % Continue with new normalized r as starting vector.
      force_reorth = 1;
      if delta>0
	fro = 0;    % Turn off full reorthogonalization.
      end
    end    
  elseif j<kmax & ~fro & beta(j+1)*delta < anorm*eps1,
    % If anorm*eps1/beta(j+1) > delta then  omega(j+1) will 
    % immediately exceed delta, and thus forcing a reorth. to occur at the
    % next step. The components of omega will mainly be determined
    % by the initial value and not the recurrence, and therefore we 
    % cannot tell reliably which components exceed eta => we might 
    % as well switch to full reorthogonalization to avoid trouble.
    % The user is probably trying to determine pathologically
    % small ( < sqrt(eps)*||A||_2 ) eigenvalues. 
    %    warning(['Semiorthogonality cannot be maintained at iteration ', ...
    %	  num2str(j),'. The matrix is probably ill-conditioned.', ...
    %	  ' Switching to full reorthogonalization.'])
    fro = 1;
    ierr = j;
  end
end

% Set up tridiagonal T_k in sparse matrix data structure.
T_k = spdiags([[beta(2:j);0] alpha(1:j) beta(1:j)],-1:1,j,j);
if nargout<2
  Q_k = T_k;
elseif j~=size(Q_k,2)
  Q_k = Q_k(:,1:j);
end
work = [nr np];


function [omega,omega_old] = update_omega(omega, omega_old, j, ...
    alpha,beta,eps1,anorm)
% UPDATE_OMEGA:  Update Simon's omega_recurrence for the Lanczos vectors.
%
% [omega,omega_old] = update_omega(omega, omega_old,j,eps1,alpha,beta,anorm)
% 

% Rasmus Munk Larsen, DAIMI, 1998.

% Estimate of contribution to roundoff errors from A*v 
%   fl(A*v) = A*v + f, 
% where ||f|| \approx eps1*||A||.
% For a full matrix A, a rule-of-thumb estimate is eps1 = sqrt(n)*eps.
T = eps1*anorm;
binv = 1/beta(j+1);

omega_old = omega;
% Update omega(1) using omega(0)==0.
omega_old(1)= beta(2)*omega(2)+ (alpha(1)-alpha(j))*omega(1) -  ...
    beta(j)*omega_old(1);
omega_old(1) = binv*(omega_old(1) + sign(omega_old(1))*T);
% Update remaining components.
k=2:j-2;
omega_old(k) = beta(k+1).*omega(k+1) + (alpha(k)-alpha(j)).*omega(k) ...
     + beta(k).*omega(k-1) - beta(j)*omega_old(k);
omega_old(k) = binv*(omega_old(k) + sign(omega_old(k))*T);       
omega_old(j-1) = binv*T;
% Swap omega and omega_old.
temp = omega;
omega = omega_old;
omega_old = omega;
omega(j) =  eps1;


function anorm = update_gbound(anorm,alpha,beta,j)
%UPDATE_GBOUND   Update Gerscgorin estimate of 2-norm 
%  ANORM = UPDATE_GBOUND(ANORM,ALPHA,BETA,J) updates the Gerscgorin bound
%  for the tridiagonal in the Lanczos process after the J'th step.
%  Applies Gerscgorins circles to T_K'*T_k instead of T_k itself
%  since this gives a tighter bound.

if j==1 % Apply Gerscgorin circles to T_k'*T_k to estimate || A ||_2
  i=j; 
  % scale to avoid overflow
  scale = max(abs(alpha(i)),abs(beta(i+1)));
  alpha(i) = alpha(i)/scale;
  beta(i+1) = beta(i+1)/scale;
  anorm = 1.01*scale*sqrt(alpha(i)^2+beta(i+1)^2 + abs(alpha(i)*beta(i+1)));
elseif j==2
  i=1;
  % scale to avoid overflow
  scale = max(max(abs(alpha(1:2)),max(abs(beta(2:3)))));
  alpha(1:2) = alpha(1:2)/scale;
  beta(2:3) = beta(2:3)/scale;
  
  anorm = max(anorm, scale*sqrt(alpha(i)^2+beta(i+1)^2 + ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2))));
  i=2;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1))) );
elseif j==3
  % scale to avoid overflow
  scale = max(max(abs(alpha(1:3)),max(abs(beta(2:4)))));
  alpha(1:3) = alpha(1:3)/scale;
  beta(2:4) = beta(2:4)/scale;
  i=2;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2))) );
  i=3;
  anorm = max(anorm,scale*sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1))) );
else
  % scale to avoid overflow
  %  scale = max(max(abs(alpha(j-2:j)),max(abs(beta(j-2:j+1)))));
  %  alpha(j-2:j) = alpha(j-2:j)/scale;
  %  beta(j-2:j+1) = beta(j-2:j+1)/scale;
  
  % Avoid scaling, which is slow. At j>3 the estimate is usually quite good
  % so just make sure that anorm is not made infinite by overflow.
  i = j-1;
  anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
      abs(beta(i+1)*beta(i+2)));
  if isfinite(anorm1)
    anorm = max(anorm,anorm1);
  end
  i = j;
  anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
      abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
      beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
      abs(alpha(i)*beta(i+1)));
  if isfinite(anorm1)
    anorm = max(anorm,anorm1);
  end
end


function [lambda,top,bot,err] = tqlb(alpha,beta)

% TQLB: Compute eigenvalues and top and bottom elements of
%       eigenvectors of a symmetric tridiagonal matrix T.
%
% [lambda,top,bot,err] = tqlb(alpha,beta)
%
% Input parameters:
%   alpha(1:n)   : Diagonal elements.
%   beta(2:n)    : Off-diagonal elements.
% Output parameters:
%   lambda(1:n)  : Computed eigenvalues.
%   top(1:n)     : Top elements in eigenvectors.
%   bot(1:n)     : Bottom elements in eigenvectors.
%   err          : dummy argument.


% Rasmus Munk Larsen, DAIMI, 1998


%
% This is a slow Matlab substitute for the 
% TQLB MEX-file.
%

warning('PROPACK:NotUsingMex','Using slow matlab code for tqlb.')
n = length(alpha);
T = spdiags([[beta(2:n);0] alpha(1:n) beta(1:n)],-1:1,n,n);

[V,lambda] = eig(full(T)); lambda = diag(lambda);
bot = V(end,:)';
top = V(1,:)';
err=0;


function [bnd,gap] = refinebounds(D,bnd,tol1)
%REFINEBONDS  Refines error bounds for Ritz values based on gap-structure
% 
%  bnd = refinebounds(lambda,bnd,tol1) 
%
%  Treat eigenvalues closer than tol1 as a cluster.

% Rasmus Munk Larsen, DAIMI, 1998

j = length(D);

if j<=1
  return
end
% Sort eigenvalues to use interlacing theorem correctly
[D,PERM] = sort(D);
bnd = bnd(PERM);


% Massage error bounds for very close Ritz values
eps34 = sqrt(eps*sqrt(eps));
[y,mid] = max(bnd);
for l=[-1,1]    
  for i=((j+1)-l*(j-1))/2:l:mid-l
    if abs(D(i+l)-D(i)) < eps34*abs(D(i))
      if bnd(i)>tol1 & bnd(i+l)>tol1
	bnd(i+l) = pythag(bnd(i),bnd(i+l));
	bnd(i) = 0;
      end
    end
  end
end
% Refine error bounds
gap = inf*ones(1,j);
gap(1:j-1) = min([gap(1:j-1);[D(2:j)-bnd(2:j)-D(1:j-1)]']);
gap(2:j) = min([gap(2:j);[D(2:j)-D(1:j-1)-bnd(1:j-1)]']);
gap = gap(:);
I = find(gap>bnd);
bnd(I) = bnd(I).*(bnd(I)./gap(I));

bnd(PERM) =  bnd;
function [r,normr,nre,s] = reorth(Q,r,normr,index,alpha,method)

%REORTH   Reorthogonalize a vector using iterated Gram-Schmidt
%
%   [R_NEW,NORMR_NEW,NRE] = reorth(Q,R,NORMR,INDEX,ALPHA,METHOD)
%   reorthogonalizes R against the subset of columns of Q given by INDEX. 
%   If INDEX==[] then R is reorthogonalized all columns of Q.
%   If the result R_NEW has a small norm, i.e. if norm(R_NEW) < ALPHA*NORMR,
%   then a second reorthogonalization is performed. If the norm of R_NEW
%   is once more decreased by  more than a factor of ALPHA then R is 
%   numerically in span(Q(:,INDEX)) and a zero-vector is returned for R_NEW.
%
%   If method==0 then iterated modified Gram-Schmidt is used.
%   If method==1 then iterated classical Gram-Schmidt is used.
%
%   The default value for ALPHA is 0.5. 
%   NRE is the number of reorthogonalizations performed (1 or 2).

% References: 
%  Aake Bjorck, "Numerical Methods for Least Squares Problems",
%  SIAM, Philadelphia, 1996, pp. 68-69.
%
%  J.~W. Daniel, W.~B. Gragg, L. Kaufman and G.~W. Stewart, 
%  ``Reorthogonalization and Stable Algorithms Updating the
%  Gram-Schmidt QR Factorization'', Math. Comp.,  30 (1976), no.
%  136, pp. 772-795.
%
%  B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
%  Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109

%  Rasmus Munk Larsen, DAIMI, 1998.

% Check input arguments.
warning('PROPACK:NotUsingMex','Using slow matlab code for reorth.')
if nargin<2
  error('Not enough input arguments.')
end
[n k1] = size(Q);
if nargin<3 | isempty(normr)
%  normr = norm(r);
  normr = sqrt(r'*r);
end
if nargin<4 | isempty(index)
  k=k1;
  index = [1:k]';
  simple = 1;
else
  k = length(index);
  if k==k1 & index(:)==[1:k]'
    simple = 1;
  else
    simple = 0;
  end
end
if nargin<5 | isempty(alpha)
  alpha=0.5; % This choice garanties that 
             % || Q^T*r_new - e_{k+1} ||_2 <= 2*eps*||r_new||_2,
             % cf. Kahans ``twice is enough'' statement proved in 
             % Parletts book.
end
if nargin<6 | isempty(method)
   method = 0;
end
if k==0 | n==0
  return
end
if nargout>3
  s = zeros(k,1);
end


normr_old = 0;
nre = 0;
while normr < alpha*normr_old | nre==0
  if method==1
    if simple
      t = Q'*r;
      r = r - Q*t;
    else
      t = Q(:,index)'*r;
      r = r - Q(:,index)*t;
    end
  else    
    for i=index, 
      t = Q(:,i)'*r; 
      r = r - Q(:,i)*t;
    end
  end
  if nargout>3
    s = s + t;
  end
  normr_old = normr;
%  normr = norm(r);
  normr = sqrt(r'*r);
  nre = nre + 1;
  if nre > 4
    % r is in span(Q) to full accuracy => accept r = 0 as the new vector.
    r = zeros(n,1);
    normr = 0;
    return
  end
end
function int = compute_int(mu,j,delta,eta,LL,strategy,extra)
%COMPUTE_INT:  Determine which Lanczos vectors to reorthogonalize against.
%
%      int = compute_int(mu,eta,LL,strategy,extra))
%
%   Strategy 0: Orthogonalize vectors v_{i-r-extra},...,v_{i},...v_{i+s+extra} 
%               with nu>eta, where v_{i} are the vectors with  mu>delta.
%   Strategy 1: Orthogonalize all vectors v_{r-extra},...,v_{s+extra} where
%               v_{r} is the first and v_{s} the last Lanczos vector with
%               mu > eta.
%   Strategy 2: Orthogonalize all vectors with mu > eta.
%
%   Notice: The first LL vectors are excluded since the new Lanczos
%   vector is already orthogonalized against them in the main iteration.

%  Rasmus Munk Larsen, DAIMI, 1998.

if (delta<eta)
  error('DELTA should satisfy DELTA >= ETA.')
end
switch strategy
  case 0
    I0 = find(abs(mu(1:j))>=delta);    
    if length(I0)==0
      [mm,I0] = max(abs(mu(1:j)));
    end    
    int = zeros(j,1);
    for i = 1:length(I0)
      for r=I0(i):-1:1
	if abs(mu(r))<eta | int(r)==1 
	  break;
	else
	  int(r) = 1;
	end
      end
      int(max(1,r-extra+1):r) = 1;
      for s=I0(i)+1:j
	if abs(mu(s))<eta | int(s)==1  
	  break;
	else
	  int(s) = 1;
	end
      end
      int(s:min(j,s+extra-1)) = 1;
    end
    if LL>0
      int(1:LL) = 0;
    end
    int = find(int);
  case 1
    int=find(abs(mu(1:j))>eta);
    int = max(LL+1,min(int)-extra):min(max(int)+extra,j);
  case 2
    int=find(abs(mu(1:j))>=eta);
end
int = int(:);



function y=Afunc(x)
% y=Afunc(x)
% Testfunction returning a linear operator applied to x.
% Used for testing lansvd.
%
% y = A'*x

% Rasmus Munk Larsen, DAIMI, 1998

global A MxV
y = A*x;
MxV = MxV + 1;

function y=AtAfunc(x)
% y=AtAfunc(x)
% Testfunction  defining a linear operator applied to x. 
% Used for testing laneig.
%
%       y = A'*(A*x)

% Rasmus Munk Larsen, DAIMI, 1998


global A MxV
y = A'*(A*x);
MxV = MxV + 2;
function y=Atransfunc(x)
% y=Atransfunc(x)
% Testfunction returning the transpose of a linear operator applied to x.
% Used for testing lansvd.
%
% y = A'*x

% Rasmus Munk Larsen, DAIMI, 1998

global A MxV
y = A'*x;
MxV = MxV + 1; 
function [sigma,bnd] = bdsqr(alpha,beta)

% BDSQR: Compute the singular values and bottom element of
%        the left singular vectors of a (k+1) x k lower bidiagonal 
%        matrix with diagonal alpha(1:k) and lower bidiagonal beta(1:k),
%        where length(alpha) = length(beta) = k.
%
% [sigma,bnd] = bdsqr(alpha,beta)
%
% Input parameters:
%   alpha(1:k)   : Diagonal elements.
%   beta(1:k)    : Sub-diagonal elements.
% Output parameters:
%   sigma(1:k)  : Computed eigenvalues.
%   bnd(1:k)    : Bottom elements in left singular vectors.

% Below is a very slow replacement for the BDSQR MEX-file.

warning('PROPACK:NotUsingMex','Using slow matlab code for bdsqr.')
k = length(alpha);
if min(size(alpha)') ~= 1  | min(size(beta)') ~= 1
  error('alpha and beta must be vectors')
elseif length(beta) ~= k
  error('alpha and beta must have the same lenght')
end    
B = spdiags([alpha(:),beta(:)],[0,-1],k+1,k);
[U,S,V] = svd(full(B),0);
sigma = diag(S);
bnd = U(end,1:k)';


function y = Cfunc(x)
% y=Cfunc(x)
% Testfunction  defining a linear operator applied to x. 
% Used for testing laneig.
%
%       y =     [ 0  A ]  * x
%               [ A' 0 ]

% Rasmus Munk Larsen, DAIMI, 1998


global A MxV
[m n] = size(A);
y = [A*x(m+1:end,:); A'*x(1:m,:)];
MxV = MxV + 2;



function [U,B_k,V,p,ierr,work] = lanbpro(varargin)

%LANBPRO Lanczos bidiagonalization with partial reorthogonalization.
%   LANBPRO computes the Lanczos bidiagonalization of a real 
%   matrix using the  with partial reorthogonalization. 
%
%   [U_k,B_k,V_k,R,ierr,work] = LANBPRO(A,K,R0,OPTIONS,U_old,B_old,V_old) 
%   [U_k,B_k,V_k,R,ierr,work] = LANBPRO('Afun','Atransfun',M,N,K,R0, ...
%                                       OPTIONS,U_old,B_old,V_old) 
%
%   Computes K steps of the Lanczos bidiagonalization algorithm with partial 
%   reorthogonalization (BPRO) with M-by-1 starting vector R0, producing a 
%   lower bidiagonal K-by-K matrix B_k, an N-by-K matrix V_k, an M-by-K 
%   matrix U_k and an M-by-1 vector R such that
%        A*V_k = U_k*B_k + R
%   Partial reorthogonalization is used to keep the columns of V_K and U_k
%   semiorthogonal:
%         MAX(DIAG((EYE(K) - V_K'*V_K))) <= OPTIONS.delta 
%   and 
%         MAX(DIAG((EYE(K) - U_K'*U_K))) <= OPTIONS.delta.
%
%   B_k = LANBPRO(...) returns the bidiagonal matrix only.
%
%   The first input argument is either a real matrix, or a string
%   containing the name of an M-file which applies a linear operator 
%   to the columns of a given matrix. In the latter case, the second 
%   input must be the name of an M-file which applies the transpose of 
%   the same linear operator to the columns of a given matrix,  
%   and the third and fourth arguments must be M and N, the dimensions 
%   of then problem.
%
%   The OPTIONS structure is used to control the reorthogonalization:
%     OPTIONS.delta:  Desired level of orthogonality 
%                     (default = sqrt(eps/K)).
%     OPTIONS.eta  :  Level of orthogonality after reorthogonalization 
%                     (default = eps^(3/4)/sqrt(K)).
%     OPTIONS.cgs  :  Flag for switching between different reorthogonalization
%                     algorithms:
%                      0 = iterated modified Gram-Schmidt  (default)
%                      1 = iterated classical Gram-Schmidt 
%     OPTIONS.elr  :  If OPTIONS.elr = 1 (default) then extended local
%                     reorthogonalization is enforced.
%     OPTIONS.onesided
%                  :  If OPTIONS.onesided = 0 (default) then both the left
%                     (U) and right (V) Lanczos vectors are kept 
%                     semiorthogonal. 
%                     OPTIONS.onesided = 1 then only the columns of U are
%                     are reorthogonalized.
%                     OPTIONS.onesided = -1 then only the columns of V are
%                     are reorthogonalized.
%     OPTIONS.waitbar
%                  :  The progress of the algorithm is display graphically.
%
%   If both R0, U_old, B_old, and V_old are provided, they must
%   contain a partial Lanczos bidiagonalization of A on the form
%
%        A V_old = U_old B_old + R0 .  
%
%   In this case the factorization is extended to dimension K x K by
%   continuing the Lanczos bidiagonalization algorithm with R0 as a 
%   starting vector.
%
%   The output array work contains information about the work used in
%   reorthogonalizing the u- and v-vectors.
%      work = [ RU  PU ]
%             [ RV  PV ] 
%   where
%      RU = Number of reorthogonalizations of U.
%      PU = Number of inner products used in reorthogonalizing U.
%      RV = Number of reorthogonalizations of V.
%      PV = Number of inner products used in reorthogonalizing V.

% References: 
% R.M. Larsen, Ph.D. Thesis, Aarhus University, 1998.
%
% G. H. Golub & C. F. Van Loan, "Matrix Computations",
% 3. Ed., Johns Hopkins, 1996.  Section 9.3.4.
%
% B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
% Prentice-Hall, Englewood Cliffs, NJ, 1980.
%
% H. D. Simon, ``The Lanczos algorithm with partial reorthogonalization'',
% Math. Comp. 42 (1984), no. 165, 115--142.
%

% Rasmus Munk Larsen, DAIMI, 1998.

% Check input arguments.

global LANBPRO_TRUTH
LANBPRO_TRUTH=0;

if LANBPRO_TRUTH==1
  global MU NU MUTRUE NUTRUE
  global MU_AFTER NU_AFTER MUTRUE_AFTER NUTRUE_AFTER
end

if nargin<1 | length(varargin)<2
  error('Not enough input arguments.');
end
narg=length(varargin);

A = varargin{1};
if isnumeric(A) | isstruct(A)
  if isnumeric(A)
    if ~isreal(A)
      error('A must be real')
    end  
    [m n] = size(A);
  elseif isstruct(A)
    [m n] = size(A.R);
  end
  k=varargin{2};
  if narg >= 3 & ~isempty(varargin{3});
    p = varargin{3};
  else
    p = rand(m,1)-0.5;
  end
  if narg < 4, options = []; else options=varargin{4}; end
  if narg > 4 
    if narg<7
      error('All or none of U_old, B_old and V_old must be provided.')
    else
      U = varargin{5}; B_k = varargin{6}; V = varargin{7};
    end
  else
    U = []; B_k = []; V = [];
  end
  if narg > 7, anorm=varargin{8}; else anorm = []; end
else
  if narg<5
    error('Not enough input arguments.');
  end
  Atrans = varargin{2};
  if ~isstr(Atrans)
    error('Afunc and Atransfunc must be names of m-files')
  end
  m = varargin{3};
  n = varargin{4};
  if ~isreal(n) | abs(fix(n)) ~= n | ~isreal(m) | abs(fix(m)) ~= m
    error('M and N must be positive integers.')
  end
  k=varargin{5};
  if narg < 6, p = rand(m,1)-0.5; else p=varargin{6}; end  
  if narg < 7, options = []; else options=varargin{7}; end  
  if narg > 7
    if  narg < 10
      error('All or none of U_old, B_old and V_old must be provided.')
    else
      U = varargin{8}; B_k = varargin{9}; V = varargin{10};
    end
  else
    U = []; B_k = []; V=[];
  end
  if narg > 10, anorm=varargin{11}; else anorm = [];  end
end

% Quick return for min(m,n) equal to 0 or 1.
if min(m,n) == 0
   U = [];  B_k = [];  V = [];  p = [];  ierr = 0;  work = zeros(2,2);
   return
elseif  min(m,n) == 1
  if isnumeric(A)
    U = 1;  B_k = A;  V = 1;  p = 0; ierr = 0; work = zeros(2,2);
  else
    U = 1;  B_k = feval(A,1); V = 1; p = 0; ierr = 0; work = zeros(2,2);
  end
  if nargout<3
    U = B_k;
  end
  return
end

% Set options.  
%m2 = 3/2*(sqrt(m)+1);
%n2 = 3/2*(sqrt(n)+1);
m2 = 3/2;
n2 = 3/2;
delta = sqrt(eps/k); % Desired level of orthogonality.
eta = eps^(3/4)/sqrt(k);    % Level of orth. after reorthogonalization.
cgs = 0;             % Flag for switching between iterated MGS and CGS.
elr = 2;             % Flag for switching extended local 
                     % reorthogonalization on and off.
gamma = 1/sqrt(2);   % Tolerance for iterated Gram-Schmidt.
onesided = 0; t = 0; waitb = 0;

% Parse options struct
if ~isempty(options) & isstruct(options)
  c = fieldnames(options);
  for i=1:length(c)
    if strmatch(c(i),'delta'), delta = getfield(options,'delta');  end
    if strmatch(c(i),'eta'), eta = getfield(options,'eta'); end
    if strmatch(c(i),'cgs'), cgs = getfield(options,'cgs'); end
    if strmatch(c(i),'elr'), elr = getfield(options,'elr'); end
    if strmatch(c(i),'gamma'), gamma = getfield(options,'gamma'); end
    if strmatch(c(i),'onesided'), onesided = getfield(options,'onesided'); end
    if strmatch(c(i),'waitbar'), waitb=1; end
  end
end

if waitb
  waitbarh = waitbar(0,'Lanczos bidiagonalization in progress...');
end

if isempty(anorm)
  anorm = []; est_anorm=1; 
else
  est_anorm=0; 
end

% Conservative statistical estimate on the size of round-off terms. 
% Notice that {\bf u} == eps/2.
FUDGE = 1.01; % Fudge factor for ||A||_2 estimate.

npu = 0; npv = 0; ierr = 0;
p = p(:);
% Prepare for Lanczos iteration.
if isempty(U)
  V = zeros(n,k); U = zeros(m,k);
  beta = zeros(k+1,1); alpha = zeros(k,1);
  beta(1) = norm(p);
  % Initialize MU/NU-recurrences for monitoring loss of orthogonality.
  nu = zeros(k,1); mu = zeros(k+1,1);
  mu(1)=1; nu(1)=1;
  
  numax = zeros(k,1); mumax = zeros(k,1);
  force_reorth = 0;  nreorthu = 0; nreorthv = 0;
  j0 = 1;
else
  j = size(U,2); % Size of existing factorization
  % Allocate space for Lanczos vectors
  U = [U, zeros(m,k-j)];
  V = [V, zeros(n,k-j)];
  alpha = zeros(k+1,1);  beta = zeros(k+1,1);
  alpha(1:j) = diag(B_k); if j>1 beta(2:j) = diag(B_k,-1); end
  beta(j+1) = norm(p);
  % Reorthogonalize p.
  if j<k & beta(j+1)*delta < anorm*eps,
    fro = 1;
    ierr = j;
  end
  int = [1:j]';
  [p,beta(j+1),rr] = reorth(U,p,beta(j+1),int,gamma,cgs);
  npu =  rr*j;  nreorthu = 1;  force_reorth= 1;  

  % Compute Gerscgorin bound on ||B_k||_2
  if est_anorm
    anorm = FUDGE*sqrt(norm(B_k'*B_k,1));
  end
  mu = m2*eps*ones(k+1,1); nu = zeros(k,1);
  numax = zeros(k,1); mumax = zeros(k,1);
  force_reorth = 1;  nreorthu = 0; nreorthv = 0;
  j0 = j+1;
end


if isnumeric(A)
  At = A';
end

if delta==0
  fro = 1; % The user has requested full reorthogonalization.
else
  fro = 0;
end

if LANBPRO_TRUTH==1
  MUTRUE = zeros(k,k); NUTRUE = zeros(k-1,k-1);
  MU = zeros(k,k); NU = zeros(k-1,k-1);
  
  MUTRUE_AFTER = zeros(k,k); NUTRUE_AFTER = zeros(k-1,k-1);
  MU_AFTER = zeros(k,k); NU_AFTER = zeros(k-1,k-1);
end

% Perform Lanczos bidiagonalization with partial reorthogonalization.
for j=j0:k
  if waitb
    waitbar(j/k,waitbarh)
  end

  if beta(j) ~= 0
    U(:,j) = p/beta(j);
  else
    U(:,j) = p;
  end

  % Replace norm estimate with largest Ritz value.
  if j==6
    B = [[diag(alpha(1:j-1))+diag(beta(2:j-1),-1)]; ...
      [zeros(1,j-2),beta(j)]];
    anorm = FUDGE*norm(B);
    est_anorm = 0;
  end
  
  %%%%%%%%%% Lanczos step to generate v_j. %%%%%%%%%%%%%
  if j==1
    if isnumeric(A)
      r = At*U(:,1);    
    elseif isstruct(A)
      r = A.R\U(:,1);          
    else
      r = feval(Atrans,U(:,1));
    end
    alpha(1) = norm(r);
    if est_anorm
      anorm = FUDGE*alpha(1);
    end
  else    
    if isnumeric(A)
      r = At*U(:,j) - beta(j)*V(:,j-1);
    elseif isstruct(A)
      r = A.R\U(:,j) - beta(j)*V(:,j-1);      
    else
      r = feval(Atrans,U(:,j))  - beta(j)*V(:,j-1);
    end
    alpha(j) = norm(r); 

    % Extended local reorthogonalization    
    if alpha(j)<gamma*beta(j) & elr & ~fro
      normold = alpha(j);
      stop = 0;
      while ~stop
	t = V(:,j-1)'*r;
	r = r - V(:,j-1)*t;
	alpha(j) = norm(r);
	if beta(j) ~= 0
	  beta(j) = beta(j) + t;
	end
	if alpha(j)>=gamma*normold
	  stop = 1;
	else
	  normold = alpha(j);
	end
      end
    end

    if est_anorm
      if j==2
	anorm = max(anorm,FUDGE*sqrt(alpha(1)^2+beta(2)^2+alpha(2)*beta(2)));
      else	
	anorm = max(anorm,FUDGE*sqrt(alpha(j-1)^2+beta(j)^2+alpha(j-1)* ...
	    beta(j-1) + alpha(j)*beta(j)));
      end			     
    end
    
    if ~fro & alpha(j) ~= 0
      % Update estimates of the level of orthogonality for the
      %  columns 1 through j-1 in V.
      nu = update_nu(nu,mu,j,alpha,beta,anorm);
      numax(j) = max(abs(nu(1:j-1)));
    end

    if j>1 & LANBPRO_TRUTH
      NU(1:j-1,j-1) = nu(1:j-1);
      NUTRUE(1:j-1,j-1) = V(:,1:j-1)'*r/alpha(j);
    end
    
    if elr>0
      nu(j-1) = n2*eps;
    end
    
    % IF level of orthogonality is worse than delta THEN 
    %    Reorthogonalize v_j against some previous  v_i's, 0<=i<j.
    if onesided~=-1 & ( fro | numax(j) > delta | force_reorth ) & alpha(j)~=0
      % Decide which vectors to orthogonalize against:
      if fro | eta==0
	int = [1:j-1]';
      elseif force_reorth==0
	int = compute_int(nu,j-1,delta,eta,0,0,0);
      end
      % Else use int from last reorth. to avoid spillover from mu_{j-1} 
      % to nu_j.
      
      % Reorthogonalize v_j 
      [r,alpha(j),rr] = reorth(V,r,alpha(j),int,gamma,cgs);
      npv = npv + rr*length(int); % number of inner products.
      nu(int) = n2*eps;  % Reset nu for orthogonalized vectors.

      % If necessary force reorthogonalization of u_{j+1} 
      % to avoid spillover
      if force_reorth==0 
	force_reorth = 1; 
      else
	force_reorth = 0; 
      end
      nreorthv = nreorthv + 1;
    end
  end

  
  % Check for convergence or failure to maintain semiorthogonality
  if alpha(j) < max(n,m)*anorm*eps & j<k, 
    % If alpha is "small" we deflate by setting it
    % to 0 and attempt to restart with a basis for a new 
    % invariant subspace by replacing r with a random starting vector:
    %j
    %disp('restarting, alpha = 0')
    alpha(j) = 0;
    bailout = 1;
    for attempt=1:3    
      r = rand(m,1)-0.5;  
      if isnumeric(A)
	r = At*r;    
      elseif isstruct(A)
	r = A.R\r;    
      else
	r = feval(Atrans,r);
      end
      nrm=sqrt(r'*r); % not necessary to compute the norm accurately here.
      int = [1:j-1]';
      [r,nrmnew,rr] = reorth(V,r,nrm,int,gamma,cgs);
      npv = npv + rr*length(int(:));        nreorthv = nreorthv + 1;
      nu(int) = n2*eps;
      if nrmnew > 0
	% A vector numerically orthogonal to span(Q_k(:,1:j)) was found. 
	% Continue iteration.
	bailout=0;
	break;
      end
    end
    if bailout
      j = j-1;
      ierr = -j;
      break;
    else
      r=r/nrmnew; % Continue with new normalized r as starting vector.
      force_reorth = 1;
      if delta>0
	fro = 0;    % Turn off full reorthogonalization.
      end
    end       
  elseif  j<k & ~fro & anorm*eps > delta*alpha(j)
%    fro = 1;
    ierr = j;
  end

  if j>1 & LANBPRO_TRUTH
    NU_AFTER(1:j-1,j-1) = nu(1:j-1);
    NUTRUE_AFTER(1:j-1,j-1) = V(:,1:j-1)'*r/alpha(j);
  end

  
  if alpha(j) ~= 0
    V(:,j) = r/alpha(j);
  else
    V(:,j) = r;
  end

  %%%%%%%%%% Lanczos step to generate u_{j+1}. %%%%%%%%%%%%%
  if waitb
    waitbar((2*j+1)/(2*k),waitbarh)
  end
  
  if isnumeric(A)
    p = A*V(:,j) - alpha(j)*U(:,j);
  elseif isstruct(A)
    p = A.Rt\V(:,j) - alpha(j)*U(:,j);
  else
    p = feval(A,V(:,j)) - alpha(j)*U(:,j);
  end
  beta(j+1) = norm(p);
  % Extended local reorthogonalization
  if beta(j+1)<gamma*alpha(j) & elr & ~fro
    normold = beta(j+1);
    stop = 0;
    while ~stop
      t = U(:,j)'*p;
      p = p - U(:,j)*t;
      beta(j+1) = norm(p);
      if alpha(j) ~= 0 
	alpha(j) = alpha(j) + t;
      end
      if beta(j+1) >= gamma*normold
	stop = 1;
      else
	normold = beta(j+1);
      end
    end
  end

  if est_anorm
    % We should update estimate of ||A||  before updating mu - especially  
    % important in the first step for problems with large norm since alpha(1) 
    % may be a severe underestimate!  
    if j==1
      anorm = max(anorm,FUDGE*pythag(alpha(1),beta(2))); 
    else
      anorm = max(anorm,FUDGE*sqrt(alpha(j)^2+beta(j+1)^2 + alpha(j)*beta(j)));
    end
  end
  
  
  if ~fro & beta(j+1) ~= 0
    % Update estimates of the level of orthogonality for the columns of V.
    mu = update_mu(mu,nu,j,alpha,beta,anorm);
    mumax(j) = max(abs(mu(1:j)));  
  end

  if LANBPRO_TRUTH==1
    MU(1:j,j) = mu(1:j);
    MUTRUE(1:j,j) = U(:,1:j)'*p/beta(j+1);
  end
  
  if elr>0
    mu(j) = m2*eps;
  end
  
  % IF level of orthogonality is worse than delta THEN 
  %    Reorthogonalize u_{j+1} against some previous  u_i's, 0<=i<=j.
  if onesided~=1 & (fro | mumax(j) > delta | force_reorth) & beta(j+1)~=0
    % Decide which vectors to orthogonalize against.
    if fro | eta==0
      int = [1:j]';
    elseif force_reorth==0
      int = compute_int(mu,j,delta,eta,0,0,0); 
    else
      int = [int; max(int)+1];
    end
    % Else use int from last reorth. to avoid spillover from nu to mu.

%    if onesided~=0
%      fprintf('i = %i, nr = %i, fro = %i\n',j,size(int(:),1),fro)
%    end
    % Reorthogonalize u_{j+1} 
    [p,beta(j+1),rr] = reorth(U,p,beta(j+1),int,gamma,cgs);    
    npu = npu + rr*length(int);  nreorthu = nreorthu + 1;    

    % Reset mu to epsilon.
    mu(int) = m2*eps;    
    
    if force_reorth==0 
      force_reorth = 1; % Force reorthogonalization of v_{j+1}.
    else
      force_reorth = 0; 
    end
  end
  
  % Check for convergence or failure to maintain semiorthogonality
  if beta(j+1) < max(m,n)*anorm*eps  & j<k,     
    % If beta is "small" we deflate by setting it
    % to 0 and attempt to restart with a basis for a new 
    % invariant subspace by replacing p with a random starting vector:
    %j
    %disp('restarting, beta = 0')
    beta(j+1) = 0;
    bailout = 1;
    for attempt=1:3    
      p = rand(n,1)-0.5;  
      if isnumeric(A)
	p = A*p;
      elseif isstruct(A)
	p = A.Rt\p;
      else
	p = feval(A,p);
      end
      nrm=sqrt(p'*p); % not necessary to compute the norm accurately here.
      int = [1:j]';
      [p,nrmnew,rr] = reorth(U,p,nrm,int,gamma,cgs);
      npu = npu + rr*length(int(:));  nreorthu = nreorthu + 1;
      mu(int) = m2*eps;
      if nrmnew > 0
	% A vector numerically orthogonal to span(Q_k(:,1:j)) was found. 
	% Continue iteration.
	bailout=0;
	break;
      end
    end
    if bailout
      ierr = -j;
      break;
    else
      p=p/nrmnew; % Continue with new normalized p as starting vector.
      force_reorth = 1;
      if delta>0
	fro = 0;    % Turn off full reorthogonalization.
      end
    end       
  elseif  j<k & ~fro & anorm*eps > delta*beta(j+1) 
%    fro = 1;
    ierr = j;
  end  
  
  if LANBPRO_TRUTH==1
    MU_AFTER(1:j,j) = mu(1:j);
    MUTRUE_AFTER(1:j,j) = U(:,1:j)'*p/beta(j+1);
  end  
end
if waitb
  close(waitbarh)
end

if j<k
  k = j;
end

B_k = spdiags([alpha(1:k) [beta(2:k);0]],[0 -1],k,k);
if nargout==1
  U = B_k;
elseif k~=size(U,2) | k~=size(V,2)  
  U = U(:,1:k);
  V = V(:,1:k);
end
if nargout>5
  work = [[nreorthu,npu];[nreorthv,npv]];
end



function mu = update_mu(muold,nu,j,alpha,beta,anorm)

% UPDATE_MU:  Update the mu-recurrence for the u-vectors.
%
%   mu_new = update_mu(mu,nu,j,alpha,beta,anorm)

%  Rasmus Munk Larsen, DAIMI, 1998.

binv = 1/beta(j+1);
mu = muold;
eps1 = 100*eps/2;
if j==1
  T = eps1*(pythag(alpha(1),beta(2)) + pythag(alpha(1),beta(1)));
  T = T + eps1*anorm;
  mu(1) = T / beta(2);
else
  mu(1) = alpha(1)*nu(1) - alpha(j)*mu(1);
%  T = eps1*(pythag(alpha(j),beta(j+1)) + pythag(alpha(1),beta(1)));
  T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(1).^2+beta(1).^2));
  T = T + eps1*anorm;
  mu(1) = (mu(1) + sign(mu(1))*T) / beta(j+1);
  % Vectorized version of loop:
  if j>2
    k=2:j-1;
    mu(k) = alpha(k).*nu(k) + beta(k).*nu(k-1) - alpha(j)*mu(k);
    %T = eps1*(pythag(alpha(j),beta(j+1)) + pythag(alpha(k),beta(k)));
    T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(k).^2+beta(k).^2));
    T = T + eps1*anorm;
    mu(k) = binv*(mu(k) + sign(mu(k)).*T);
  end
%  T = eps1*(pythag(alpha(j),beta(j+1)) + pythag(alpha(j),beta(j)));
  T = eps1*(sqrt(alpha(j).^2+beta(j+1).^2) + sqrt(alpha(j).^2+beta(j).^2));
  T = T + eps1*anorm;
  mu(j) = beta(j)*nu(j-1);
  mu(j) = (mu(j) + sign(mu(j))*T) / beta(j+1);
end  
mu(j+1) = 1;


function nu = update_nu(nuold,mu,j,alpha,beta,anorm)

% UPDATE_MU:  Update the nu-recurrence for the v-vectors.
%
%  nu_new = update_nu(nu,mu,j,alpha,beta,anorm)

%  Rasmus Munk Larsen, DAIMI, 1998.

nu = nuold;
ainv = 1/alpha(j);
eps1 = 100*eps/2;
if j>1
  k = 1:(j-1);
%  T = eps1*(pythag(alpha(k),beta(k+1)) + pythag(alpha(j),beta(j)));
  T = eps1*(sqrt(alpha(k).^2+beta(k+1).^2) + sqrt(alpha(j).^2+beta(j).^2));
  T = T + eps1*anorm;
  nu(k) = beta(k+1).*mu(k+1) + alpha(k).*mu(k) - beta(j)*nu(k);
  nu(k) = ainv*(nu(k) + sign(nu(k)).*T);
end
nu(j) = 1;

function x = pythag(y,z)
%PYTHAG Computes sqrt( y^2 + z^2 ).
%
% x = pythag(y,z)
%
% Returns sqrt(y^2 + z^2) but is careful to scale to avoid overflow.

% Christian H. Bischof, Argonne National Laboratory, 03/31/89.

[m n] = size(y);
if m>1 | n>1
  y = y(:); z=z(:);
  rmax = max(abs([y z]'))';
  id=find(rmax==0);
  if length(id)>0
    rmax(id) = 1;
    x = rmax.*sqrt((y./rmax).^2 + (z./rmax).^2);
    x(id)=0;
  else
    x = rmax.*sqrt((y./rmax).^2 + (z./rmax).^2);
  end
  x = reshape(x,m,n);
else
  rmax = max(abs([y;z]));
  if (rmax==0)
    x = 0;
  else
    x = rmax*sqrt((y/rmax)^2 + (z/rmax)^2);
  end
end
  
function  [rows, cols, entries, rep, field, symm] = mminfo(filename)
%
%  function  [rows, cols, entries, rep, field, symmetry] = mminfo(filename)
%
%      Reads the contents of the Matrix Market file 'filename'
%      and extracts size and storage information.
%
%      In the case of coordinate matrices, entries refers to the
%      number of coordinate entries stored in the file.  The number
%      of non-zero entries in the final matrix cannot be determined
%      until the data is read (and symmetrized, if necessary).
%
%      In the case of array matrices, entries is the product
%      rows*cols, regardless of whether symmetry was used to
%      store the matrix efficiently.
%
%

mmfile = fopen(filename,'r');
if ( mmfile == -1 )
 disp(filename);
 error('File not found');
end;

header = fgets(mmfile);
if (header == -1 )
  error('Empty file.')
end
      
% NOTE: If using a version of Matlab for which strtok is not
%       defined, substitute 'gettok' for 'strtok' in the 
%       following lines, and download gettok.m from the
%       Matrix Market site.    
[head0,header]   = strtok(header);  % see note above
[head1,header]   = strtok(header);
[rep,header]     = strtok(header);
[field,header]   = strtok(header);
[symm,header]    = strtok(header);
head1 = lower(head1);
rep   = lower(rep);
field = lower(field);
symm  = lower(symm);
if ( length(symm) == 0 )
   disp('Not enough words in header line.') 
   disp('Recognized format: ')
   disp('%%MatrixMarket matrix representation field symmetry')
   error('Check header line.')
end
if ( ~ strcmp(head0,'%%MatrixMarket') )
   error('Not a valid MatrixMarket header.')
end
if (  ~ strcmp(head1,'matrix') )
   disp(['This seems to be a MatrixMarket ',head1,' file.']);
   disp('This function only knows how to read MatrixMarket matrix files.');
   disp('  ');
   error('  ');
end

% Read through comments, ignoring them

commentline = fgets(mmfile);
while length(commentline) > 0 & commentline(1) == '%',
  commentline = fgets(mmfile);
end

% Read size information, then branch according to
% sparse or dense format

if ( strcmp(rep,'coordinate')) %  read matrix given in sparse 
                              %  coordinate matrix format

  [sizeinfo,count] = sscanf(commentline,'%d%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d%d');
     if ( count > 0 & count ~= 3 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = sizeinfo(3);

elseif ( strcmp(rep,'array') ) %  read matrix given in dense 
                               %  array (column major) format

  [sizeinfo,count] = sscanf(commentline,'%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d');
     if ( count > 0 & count ~= 2 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = rows*cols;
end

fclose(mmfile);
% Done.

function  [A,rows,cols,entries,rep,field,symm] = mmread(filename)
%
% function  [A] = mmread(filename)
%
% function  [A,rows,cols,entries,rep,field,symm] = mmread(filename)
%
%      Reads the contents of the Matrix Market file 'filename'
%      into the matrix 'A'.  'A' will be either sparse or full,
%      depending on the Matrix Market format indicated by
%      'coordinate' (coordinate sparse storage), or
%      'array' (dense array storage).  The data will be duplicated
%      as appropriate if symmetry is indicated in the header.
%
%      Optionally, size information about the matrix can be 
%      obtained by using the return values rows, cols, and
%      entries, where entries is the number of nonzero entries
%      in the final matrix. Type information can also be retrieved
%      using the optional return values rep (representation), field,
%      and symm (symmetry).
%

mmfile = fopen(filename,'r');
if ( mmfile == -1 )
 disp(filename);
 error('File not found');
end;

header = fgets(mmfile);
if (header == -1 )
  error('Empty file.')
end

% NOTE: If using a version of Matlab for which strtok is not
%       defined, substitute 'gettok' for 'strtok' in the 
%       following lines, and download gettok.m from the
%       Matrix Market site.    
[head0,header]   = strtok(header);  % see note above
[head1,header]   = strtok(header);
[rep,header]     = strtok(header);
[field,header]   = strtok(header);
[symm,header]    = strtok(header);
head1 = lower(head1);
rep   = lower(rep);
field = lower(field);
symm  = lower(symm);
if ( length(symm) == 0 )
   disp(['Not enough words in header line of file ',filename]) 
   disp('Recognized format: ')
   disp('%%MatrixMarket matrix representation field symmetry')
   error('Check header line.')
end
if ( ~ strcmp(head0,'%%MatrixMarket') )
   error('Not a valid MatrixMarket header.')
end
if (  ~ strcmp(head1,'matrix') )
   disp(['This seems to be a MatrixMarket ',head1,' file.']);
   disp('This function only knows how to read MatrixMarket matrix files.');
   disp('  ');
   error('  ');
end

% Read through comments, ignoring them

commentline = fgets(mmfile);
while length(commentline) > 0 & commentline(1) == '%',
  commentline = fgets(mmfile);
end

% Read size information, then branch according to
% sparse or dense format

if ( strcmp(rep,'coordinate')) %  read matrix given in sparse 
                              %  coordinate matrix format

  [sizeinfo,count] = sscanf(commentline,'%d%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d%d');
     if ( count > 0 & count ~= 3 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = sizeinfo(3);
  
  if  ( strcmp(field,'real') )               % real valued entries:
  
    [T,count] = fscanf(mmfile,'%f',3);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 3*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,3,entries)';
    A = sparse(T(:,1), T(:,2), T(:,3), rows , cols);
  
  elseif   ( strcmp(field,'complex'))            % complex valued entries:
  
    T = fscanf(mmfile,'%f',4);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 4*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,4,entries)';
    A = sparse(T(:,1), T(:,2), T(:,3) + T(:,4)*sqrt(-1), rows , cols);
  
  elseif  ( strcmp(field,'pattern'))    % pattern matrix (no values given):
  
    T = fscanf(mmfile,'%f',2);
    T = [T; fscanf(mmfile,'%f')];
    if ( size(T) ~= 2*entries )
       message = ...
       str2mat('Data file does not contain expected amount of data.',...
               'Check that number of data lines matches nonzero count.');
       disp(message);
       error('Invalid data.');
    end
    T = reshape(T,2,entries)';
    A = sparse(T(:,1), T(:,2), ones(entries,1) , rows , cols);

  end

elseif ( strcmp(rep,'array') ) %  read matrix given in dense 
                               %  array (column major) format

  [sizeinfo,count] = sscanf(commentline,'%d%d');
  while ( count == 0 )
     commentline =  fgets(mmfile);
     if (commentline == -1 )
       error('End-of-file reached before size information was found.')
     end
     [sizeinfo,count] = sscanf(commentline,'%d%d');
     if ( count > 0 & count ~= 2 )
       error('Invalid size specification line.')
     end
  end
  rows = sizeinfo(1);
  cols = sizeinfo(2);
  entries = rows*cols;
  if  ( strcmp(field,'real') )               % real valued entries:
    A = fscanf(mmfile,'%f',1);
    A = [A; fscanf(mmfile,'%f')];
    if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') ) 
      for j=1:cols-1,
        currenti = j*rows;
        A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
      end
    elseif ( ~ strcmp(symm,'general') )
      disp('Unrecognized symmetry')
      disp(symm)
      disp('Recognized choices:')
      disp('   symmetric')
      disp('   hermitian')
      disp('   skew-symmetric')
      disp('   general')
      error('Check symmetry specification in header.');
    end
      A = reshape(A,rows,cols);
  elseif  ( strcmp(field,'complex'))         % complx valued entries:
    tmpr = fscanf(mmfile,'%f',1);
    tmpi = fscanf(mmfile,'%f',1);
    A  = tmpr+tmpi*i;
    for j=1:entries-1
      tmpr = fscanf(mmfile,'%f',1);
      tmpi = fscanf(mmfile,'%f',1);
      A  = [A; tmpr + tmpi*i];
    end
    if ( strcmp(symm,'symmetric') | strcmp(symm,'hermitian') | strcmp(symm,'skew-symmetric') ) 
      for j=1:cols-1,
        currenti = j*rows;
        A = [A(1:currenti); zeros(j,1);A(currenti+1:length(A))];
      end
    elseif ( ~ strcmp(symm,'general') )
      disp('Unrecognized symmetry')
      disp(symm)
      disp('Recognized choices:')
      disp('   symmetric')
      disp('   hermitian')
      disp('   skew-symmetric')
      disp('   general')
      error('Check symmetry specification in header.');
    end
    A = reshape(A,rows,cols);
  elseif  ( strcmp(field,'pattern'))    % pattern (makes no sense for dense)
   disp('Matrix type:',field)
   error('Pattern matrix type invalid for array storage format.');
  else                                 % Unknown matrix type
   disp('Matrix type:',field)
   error('Invalid matrix type specification. Check header against MM documentation.');
  end
end

%
% If symmetric, skew-symmetric or Hermitian, duplicate lower
% triangular part and modify entries as appropriate:
%

if ( strcmp(symm,'symmetric') )
   A = A + A.' - diag(diag(A));
   entries = nnz(A);
elseif ( strcmp(symm,'hermitian') )
   A = A + A' - diag(diag(A));
   entries = nnz(A);
elseif ( strcmp(symm,'skew-symmetric') )
   A = A - A';
   entries = nnz(A);
end

fclose(mmfile);
% Done.

function [ err ] = mmwrite(filename,A,comment,field,precision)
%
% Function: mmwrite(filename,A,comment,field,precision)
%
%    Writes the sparse or dense matrix A to a Matrix Market (MM) 
%    formatted file.
%
% Required arguments: 
%
%                 filename  -  destination file
%
%                 A         -  sparse or full matrix
%
% Optional arguments: 
%
%                 comment   -  matrix of comments to prepend to
%                              the MM file.  To build a comment matrix,
%                              use str2mat. For example:
%
%                              comment = str2mat(' Comment 1' ,...
%                                                ' Comment 2',...
%                                                ' and so on.',...
%                                                ' to attach a date:',...
%                                                [' ',date]);
%                              If ommitted, a single line date stamp comment
%                              will be included.
%
%                 field     -  'real'
%                              'complex'
%                              'integer'
%                              'pattern'
%                              If ommitted, data will determine type.
%
%                 precision -  number of digits to display for real 
%                              or complex values
%                              If ommitted, full working precision is used.
%

if ( nargin == 5) 
  precision = 16;
elseif ( nargin == 4) 
  precision = 16;
elseif ( nargin == 3) 
  mattype = 'real'; % placeholder, will check after FIND-ing A
  precision = 16;
elseif ( nargin == 2) 
  comment = '';
  % Check whether there is an imaginary part:
  mattype = 'real'; % placeholder, will check after FIND-ing A
  precision = 16;
end

mmfile = fopen([filename],'w');
if ( mmfile == -1 )
 error('Cannot open file for output');
end;


[M,N] = size(A);

%%%%%%%%%%%%%       This part for sparse matrices     %%%%%%%%%%%%%%%%
if ( issparse(A) )

  [I,J,V] = find(A);
  if ( sum(abs(imag(nonzeros(V)))) > 0 )
    Vreal = 0; 
  else 
    Vreal = 1; 
  end

  if ( ~ strcmp(mattype,'pattern') & Vreal )
    mattype = 'real'; 
  elseif ( ~ strcmp(mattype,'pattern') )
    mattype = 'complex';
  end
%
% Determine symmetry:
%
  if ( M ~= N )
    symm = 'general';
    issymm = 0;
    NZ = length(V);
  else
    issymm = 1;
    NZ = length(V);
    for i=1:NZ
      if ( A(J(i),I(i)) ~= V(i) )
        issymm = 0;
        break;
      end
    end
    if ( issymm )
      symm = 'symmetric';
      ATEMP = tril(A);
      [I,J,V] = find(ATEMP);
      NZ = nnz(ATEMP);
    else
      isskew = 1;
      for i=1:NZ
        if ( A(J(i),I(i)) ~= - V(i) )
          isskew = 0;
          break;
        end
      end
      if ( isskew )
        symm = 'skew-symmetric';
        ATEMP = tril(A);
        [I,J,V] = find(ATEMP);
        NZ = nnz(ATEMP);
      elseif ( strcmp(mattype,'complex') )
        isherm = 1;
        for i=1:NZ
          if ( A(J(i),I(i)) ~= conj(V(i)) )
            isherm = 0;
            break;
          end
        end
        if ( isherm )
          symm = 'hermitian';
          ATEMP = tril(A);
          [I,J,V] = find(ATEMP);
          NZ = nnz(ATEMP);
        else 
          symm = 'general';
          NZ = nnz(A);
        end
      else
        symm = 'general';
        NZ = nnz(A);
      end
    end
  end

% Sparse coordinate format:

  rep = 'coordinate';


  fprintf(mmfile,'%%%%MatrixMarket matrix %s %s %s\n',rep,mattype,symm);
  [MC,NC] = size(comment);
  if ( MC == 0 )
    fprintf(mmfile,'%% Generated %s\n',[date]);
  else
    for i=1:MC,
      fprintf(mmfile,'%%%s\n',comment(i,:));
    end
  end
  fprintf(mmfile,'%d %d %d\n',M,N,NZ);
  cplxformat = sprintf('%%d %%d %% .%dg %% .%dg\n',precision,precision);
  realformat = sprintf('%%d %%d %% .%dg\n',precision);
  if ( strcmp(mattype,'real') )
     for i=1:NZ
        fprintf(mmfile,realformat,I(i),J(i),V(i));
     end;
  elseif ( strcmp(mattype,'complex') )
  for i=1:NZ
     fprintf(mmfile,cplxformat,I(i),J(i),real(V(i)),imag(V(i)));
  end;
  elseif ( strcmp(mattype,'pattern') )
     for i=1:NZ
        fprintf(mmfile,'%d %d\n',I(i),J(i));
     end;
  else  
     err = -1;
     disp('Unsupported mattype:')
     mattype
  end;

%%%%%%%%%%%%%       This part for dense matrices      %%%%%%%%%%%%%%%%
else
  if ( sum(abs(imag(nonzeros(A)))) > 0 )
    Areal = 0; 
  else 
    Areal = 1; 
  end
  if ( ~strcmp(mattype,'pattern') & Areal )
    mattype = 'real';
  elseif ( ~strcmp(mattype,'pattern')  )
    mattype = 'complex';
  end
%
% Determine symmetry:
%
  if ( M ~= N )
    issymm = 0;
    symm = 'general';
  else
    issymm = 1;
    for j=1:N 
      for i=j+1:N
        if (A(i,j) ~= A(j,i) )
          issymm = 0;   
          break; 
        end
      end
      if ( ~ issymm ) break; end
    
    end
    if ( issymm )
      symm = 'symmetric';
    else
      isskew = 1;
      for j=1:N 
        for i=j+1:N
          if (A(i,j) ~= - A(j,i) )
            isskew = 0;   
            break; 
          end
        end
        if ( ~ isskew ) break; end
      end
      if ( isskew )
        symm = 'skew-symmetric';
      elseif ( strcmp(mattype,'complex') )
        isherm = 1;
        for j=1:N 
          for i=j+1:N
            if (A(i,j) ~= conj(A(j,i)) )
              isherm = 0;   
              break; 
            end
          end
          if ( ~ isherm ) break; end
        end
        if ( isherm )
          symm = 'hermitian';
        else 
          symm = 'general';
        end
      else
        symm = 'general';
      end
    end
  end

% Dense array format:

  rep = 'array';
  [MC,NC] = size(comment);
  fprintf(mmfile,'%%%%MatrixMarket mtx %s %s %s\n',rep,mattype,symm);
  for i=1:MC,
    fprintf(mmfile,'%%%s\n',comment(i,:));
  end;
  fprintf(mmfile,'%d %d\n',M,N);
  cplxformat = sprintf('%% .%dg %% .%dg\n', precision,precision);
  realformat = sprintf('%% .%dg\n', precision);
  if ( ~ strcmp(symm,'general') )
     rowloop = 'j';
  else 
     rowloop = '1';
  end
  if ( strcmp(mattype,'real') )
     for j=1:N
       for i=eval(rowloop):M
          fprintf(mmfile,realformat,A(i,j));
       end
     end
  elseif ( strcmp(mattype,'complex') )
     for j=1:N
       for i=eval(rowloop):M
          fprintf(mmfile,cplxformat,real(A(i,j)),imag(A(i,j)));
       end
     end
  elseif ( strcmp(mattype,'pattern') )
     err = -2
     disp('Pattern type inconsistant with dense matrix')
  else
     err = -2
     disp('Unknown matrix type:')
     mattype
  end
end

fclose(mmfile);


% function anorm = update_gbound(anorm,alpha,beta,j)
% %UPDATE_GBOUND   Update Gerscgorin estimate of 2-norm 
% %  ANORM = UPDATE_GBOUND(ANORM,ALPHA,BETA,J) updates the Gersgorin bound
% %  for the tridiagonal in the Lanczos process after the J'th step.
% %  Applies Gerscgorins circles to T_K'*T_k instead of T_k itself
% %  since this gives a tighter bound.
% 
% if j==1 % Apply Gerscgorin circles to T_k'*T_k to estimate || A ||_2
%   i=j; 
%   % scale to avoid overflow
%   scale = max(abs(alpha(i)),abs(beta(i+1)));
%   alpha(i) = alpha(i)/scale;
%   beta(i) = beta(i)/scale;
%   anorm = 1.01*scale*sqrt(alpha(i)^2+beta(i+1)^2 + abs(alpha(i)*beta(i+1)));
% elseif j==2
%   i=1;
%   % scale to avoid overflow
%   scale = max(max(abs(alpha(1:2)),max(abs(beta(2:3)))));
%   alpha(1:2) = alpha(1:2)/scale;
%   beta(2:3) = beta(2:3)/scale;
%   
%   anorm = max(anorm, scale*sqrt(alpha(i)^2+beta(i+1)^2 + ...
%       abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
%       abs(beta(i+1)*beta(i+2))));
%   i=2;
%   anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
%       beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
%       abs(alpha(i)*beta(i+1))) );
% elseif j==3
%   % scale to avoid overflow
%   scale = max(max(abs(alpha(1:3)),max(abs(beta(2:4)))));
%   alpha(1:3) = alpha(1:3)/scale;
%   beta(2:4) = beta(2:4)/scale;
%   i=2;
%   anorm = max(anorm,scale*sqrt(abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
%       beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
%       abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
%       abs(beta(i+1)*beta(i+2))) );
%   i=3;
%   anorm = max(anorm,scale*sqrt(abs(beta(i)*beta(i-1)) + ...
%       abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
%       beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
%       abs(alpha(i)*beta(i+1))) );
% else
%   % scale to avoid overflow
%   %  scale = max(max(abs(alpha(j-2:j)),max(abs(beta(j-2:j+1)))));
%   %  alpha(j-2:j) = alpha(j-2:j)/scale;
%   %  beta(j-2:j+1) = beta(j-2:j+1)/scale;
%   
%   % Avoid scaling, which is slow. At j>3 the estimate is usually quite good
%   % so just make sure that anorm is not made infinite by overflow.
%   i = j-1;
%   anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
%       abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
%       beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
%       abs(alpha(i)*beta(i+1) + alpha(i+1)*beta(i+1)) + ...
%       abs(beta(i+1)*beta(i+2)));
%   if isfinite(anorm1)
%     anorm = max(anorm,anorm1);
%   end
%   i = j;
%   anorm1 = sqrt(abs(beta(i)*beta(i-1)) + ...
%       abs(beta(i)*alpha(i-1) + alpha(i)*beta(i)) + ...
%       beta(i)^2+alpha(i)^2+beta(i+1)^2 +  ...
%       abs(alpha(i)*beta(i+1)));
%   if isfinite(anorm1)
%     anorm = max(anorm,anorm1);
%   end
% end
% function [bnd,gap] = refinebounds(D,bnd,tol1)
% %REFINEBONDS  Refines error bounds for Ritz values based on gap-structure
% % 
% %  bnd = refinebounds(lambda,bnd,tol1) 
% %
% %  Treat eigenvalues closer than tol1 as a cluster.
% 
% % Rasmus Munk Larsen, DAIMI, 1998
% 
% j = length(D);
% 
% if j<=1
%   return
% end
% % Sort eigenvalues to use interlacing theorem correctly
% [D,PERM] = sort(D);
% bnd = bnd(PERM);
% 
% 
% % Massage error bounds for very close Ritz values
% eps34 = sqrt(eps*sqrt(eps));
% [y,mid] = max(bnd);
% for l=[-1,1]    
%   for i=((j+1)-l*(j-1))/2:l:mid-l
%     if abs(D(i+l)-D(i)) < eps34*abs(D(i))
%       if bnd(i)>tol1 & bnd(i+l)>tol1
% 	bnd(i+l) = pythag(bnd(i),bnd(i+l));
% 	bnd(i) = 0;
%       end
%     end
%   end
% end
% % Refine error bounds
% gap = inf*ones(1,j);
% gap(1:j-1) = min([gap(1:j-1);[D(2:j)-bnd(2:j)-D(1:j-1)]']);
% gap(2:j) = min([gap(2:j);[D(2:j)-D(1:j-1)-bnd(1:j-1)]']);
% gap = gap(:);
% I = find(gap>bnd);
% bnd(I) = bnd(I).*(bnd(I)./gap(I));
% 
% bnd(PERM) =  bnd;