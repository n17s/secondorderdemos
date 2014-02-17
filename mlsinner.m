function w=mlsinner(trainx,yt,lambda,initial,weights)
    % Inner loop for Multinomial logistic regression
    [n,k]=size(yt);
    [~,d]=size(trainx);
    rx=bsxfun(@times,trainx,sqrt(weights));
    C=chol((rx'*rx)+lambda*sqrt(n)*eye(d),'lower');
    u=zeros(d,k);
    w=u;   
    li=1;
    linext=1;
    for i=1:8
        pt=trainx*u+initial;
        [z,~]=max(pt');
        q=exp(bsxfun(@minus,pt,z'));% numerically stable exp
        pp=q./repmat(sum(q,2),1,k); % probabilistic predictions
        g=trainx'*(yt-pp);          % gradient
        wold=w;                    
        w=u+C'\(C\g);          
        gi=(1-li)/linext;
        u=(1-gi)*w+gi*wold;
        li=linext;
        linext=(1+sqrt(1+4*li^2))/2;% accelerated gradient update
    end
end
