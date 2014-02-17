function w=mls(trainx,yt,lambda)
    % Multinomial logistic regression via least squares
    [n,k]=size(yt);
    [~,d]=size(trainx);
    %compute preconditioner; a subsample should suffice
    sss=2*d; %sub-sample size
    rp=randperm(n,sss);
    smallx=trainx(rp,:);
    C=chol(0.5*n/sss*(smallx'*smallx)+lambda*sqrt(sss)*eye(d),'lower');
    %true preconditioner; 
    %C=chol(0.5*(trainx'*trainx)+lambda*sqrt(n)*eye(d),'lower');
    
    %initialize accelerated gradient variables
    u=zeros(d,k);
    w=u;
    %do not mess with these   
    li=1;
    linext=1;
    for i=1:100
        pt=trainx*u;
        [z,~]=max(pt');
        q=exp(bsxfun(@minus,pt,z'));% numerically stable exp
        pp=q./repmat(sum(q,2),1,k); % probabilistic predictions
        g=trainx'*(yt-pp);          % gradient
        normg=norm(g,'fro')/sqrt(d*k*n);
        if normg<0.01
            break
        end
        %accelerated gradient updates
        wold=w;                    
        w=u+C'\(C\g);          
        gi=(1-li)/linext;
        u=(1-gi)*w+gi*wold;
        li=linext;
        linext=(1+sqrt(1+4*li^2))/2;
    end
end
