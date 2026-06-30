function y = getCurveletFeatureVector(Curvelets, route)
%Gets the feature vector from the Curvelet coefficient structure, those
%features are the parameters beta, alpha and mu of the normalized gaussian
%distribution
%   Detailed explanation goes here
    featureVector=[];
    %matriz_resultados = {};
    [~,sc]=size(Curvelets);
    for i=1:sc
        scale=Curvelets{i};
        [~, ss]=size(scale);
        for j=1:ss
             coefficientMatrix = scale{j};
             distributionData=real(coefficientMatrix(:));
             
             %% Check distribution of curvelet coefficients and save it
             %f = figure('Visible', 'off');
             %hist(distributionData, 40);
             %hist_max = max(hist(distributionData, 40));
             %saveas(f, [route '_scale_' num2str(i) '_subband_' num2str(j) '.png']);
             
             [alpha,beta]=ggmle(distributionData);
             mu=mean(distributionData);

             features=[alpha,beta,mu];
             featureVector = [featureVector features];
        end
    end
    
    y=featureVector;
    
end

