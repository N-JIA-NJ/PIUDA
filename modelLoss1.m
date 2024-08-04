function [loss,gradients] = modelLoss1(net,X,T,idxMiniBatch)

    order=idxMiniBatch;

    % Forward data through the dlnetwork object.
    Y = forward(net,X);
    Y1=zeros(4,10);
    for i=1:length(Y)
    [~,y]=max(Y(:,i));
    Y1(y,i)=1;
    end

    %physics-informed
    plabel=Y1;
    loss_f=Faultfrequency(order,T,plabel);
    %function <Faultfrequency> involves other projects and will be made public after the confidentiality is lifted.
    loss_f=mapminmax(loss_f,0,1);
    loss_f=mean(loss_f);
    % Compute loss.
    loss_c = crossentropy(Y,T);
    loss=loss_c+loss_f;
    % Compute gradients.
    gradients = dlgradient(loss,net.Learnables);

end
