rand('seed',867);
randn('seed',5309);
 
tic
fprintf('loading mnist');
 
% get mnist from http://cs.nyu.edu/~roweis/data/mnist_all.mat
load('mnist_all.mat');
 
trainx=single([train0; train1; train2; train3; train4; train5; train6; train7; train8; train9])/255.0;
testx=single([test0; test1; test2; test3; test4; test5; test6; test7; test8; test9])/255.0;
st=[size(train0,1); size(train1,1); size(train2,1); size(train3,1); size(train4,1); size(train5,1); size(train6,1); size(train7,1); size(train8,1); size(train9,1)];
ss=[size(test0,1); size(test1,1); size(test2,1); size(test3,1); size(test4,1); size(test5,1); size(test6,1); size(test7,1); size(test8,1); size(test9,1)];
paren = @(x, varargin) x(varargin{:});
yt=[]; for i=1:10; yt=[yt; repmat(paren(eye(10),i,:),st(i),1)]; end
ys=[]; for i=1:10; ys=[ys; repmat(paren(eye(10),i,:),ss(i),1)]; end
 
clear i st ss
clear train0 train1 train2 train3 train4 train5 train6 train7 train8 train9
clear test0 test1 test2 test3 test4 test5 test6 test7 test8 test9
 
fprintf(' finished: ');
toc
 
tic
fprintf('preparing the data ');
 
% (uncentered) pca to 50 ... makes subsequent operations faster,
% but also makes the random projection more efficient by focusing on
% where the data is
 
opts.isreal = true; 
[v,~]=eigs(double(trainx'*trainx),50,'LM',opts);
trainx=trainx*v;
testx=testx*v; 
clear v opts;
 
% estimate kernel bandwidth using the "median trick"
% this is a standard Gaussian kernel technique
 
[n,k]=size(yt);
[m,p]=size(testx);
sz=3000;
perm=randperm(n);
sample=trainx(perm(1:sz),:);
norms=sum(sample.^2,2);
dist=norms*ones(1,sz)+ones(sz,1)*norms'-2*sample*sample';
scale=1/sqrt(median(dist(:)));
clear sz perm sample norms dist;
fprintf(' finished: ');
toc

tic
fprintf('preparing the data a little more :-) ');
d=4000;
r=randn(p,d);
b=2.0*pi*rand(1,d);
xt=cos(bsxfun(@plus,scale*trainx*r,b));
xs=cos(bsxfun(@plus,scale*testx*r,b));
 
fprintf(' finished: ');
toc

fprintf('starting calibrated regression ');
ps=cls(xt,yt,0.5,xs);
fprintf(' finished: ');
toc;
[zs,yhats]=max(ps');
[~,testy]=max(ys');
errors=sum(yhats~=testy)/m;
qs=exp(ps);
fprintf('test accuracy: %g\n',1-errors);

tic
fprintf('starting multinomial regression via least squares ');
w=mls(xt,yt,3);
ps=xs*w;
fprintf(' finished: ');
toc;
[zs,yhats]=max(ps');
[~,testy]=max(ys');
errors=sum(yhats~=testy)/m;
qs=exp(ps);
fprintf('test accuracy: %g\n',1-errors);

tic
fprintf('starting stagewise regression\n');
ps=stagewisemls(trainx,yt,2,testx,2000,20,scale);
fprintf(' finished: ');
toc
[zs,yhats]=max(ps');
[~,testy]=max(ys');
errors=sum(yhats~=testy)/m;
qs=exp(ps);
fprintf('test accuracy: %g\n',1-errors);
