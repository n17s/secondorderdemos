function ps=stagewisemls(trainx,yt,lambda,testx,batchsize,iterations,scale)
    [n,k]=size(yt);
    [m,p]=size(testx);
    
    pt=zeros(n,k);
    ps=zeros(m,k);
    t=0.5*ones(n,1);
    [~,trainy]=max(yt');
    for batch=1:iterations
        r=randn(p,batchsize);
        b=2.0*pi*rand(1,batchsize);
        trainxb=cos(bsxfun(@plus,scale*trainx*r,b));
        testxb=cos(bsxfun(@plus,scale*testx*r,b));
        w=mlsinner(trainxb,yt,2,pt,t);
        ps=ps+testxb*w; 
        pt=pt+trainxb*w;
        [zt,yhatt]=max(pt');
        errors=sum(yhatt~=trainy)/m;
        fprintf('iteration %2d,train accuracy: %g\n',batch,1-errors);
        qt=exp(bsxfun(@minus,pt,zt'));
        pp=qt./repmat(sum(qt,2),1,k); 
        t=0.1*0.5+0.9*2*(max(pp'.*(1-pp')))';
    end
end
