function [loss,gradients,TX,probabilitylabel,iiii] = modelLossT(net,X,T,idxMiniBatch, probabilitylabel)

    changdu=length(idxMiniBatch);
    dlabel=zeros(4,changdu); 
    Y1=zeros(4,changdu); 

    % Forward data through the dlnetwork object.
    Y = forward(net,X);
    loss = crossentropy(Y,T);
    % Compute gradients.
    gradients = dlgradient(loss,net.Learnables);

    %soft pseudo label
    order=idxMiniBatch;
    pblabel=probabilitylabel(order,:);%probability label
    pblabel=pblabel';
    PP=1./pblabel;

    %one-hot of soft pseudo label
    for p=1:length(order)
        [~,m]=min(PP(:,p));
        dlabel(m,p)=1;
    end
    
    %Predicted Label
     YY=Y+1;
     for y=1:length(order)
        [~,n]=max(YY(:,y));
        Y1(n,y)=1;
     end
     YY1=Y1';
tic
    %If a certain condition is met, the hard fake label will be processed
   for d=1:length(order)
       [~,a1]=max(T(:,d));%hard
       [~,b1]=max(Y1(:,d));%Predicted
       [~,c1]=max(pblabel(:,d));%soft
    if a1==b1
%        probabilitylabel(order(d),:)=T(:,d);
       TX(:,d)=T(:,d);
    end

    if a1~=b1%This indicates that the data is physically inconsistent.
        if a1==c1%Soft and hard fake labels still have a certain degree of reliability，
             %Find the maximum and its position
            [e1,f1]=max(pblabel(:,d));
            %Calculates values ​​that follow a sigmoid distribution
            ee1 = 0.1*sig(e1);
            %Find the second largest and its location
            b=pblabel(:,d);
            b(find(b==max(b)))=min(b);
            [~,f2]=max(b);
            %Changes to soft fake labels
            pblabel(f1,d)=abs(pblabel(f1,d)-2*ee1);
            pblabel(:,d)=pblabel(:,d)+ee1;
            bb=pblabel(:,d)';
            %Put the updated results into the entire matrix
            TX(:,d)=pblabel(:,d)./sum(pblabel(:,d));
            probabilitylabel(order(d),:)=bb./sum(bb);
        end

         if a1~=c1
            [e1,f1]=max(pblabel(:,d));
            ee1 = sig(e1);
            b=pblabel(:,d);
            b(find(b==max(b)))=min(b);
            [~,f2]=max(b);
            pblabel(f1,d)=abs(pblabel(f1,d)-2*ee1);
            pblabel(:,d)=pblabel(:,d)+ee1;
            bb=pblabel(:,d)';
            TX(:,d)=pblabel(:,d)./sum(pblabel(:,d));
            probabilitylabel(order(d),:)=bb./sum(bb);
         end
    end

   end
toc
iiii=toc;
end

