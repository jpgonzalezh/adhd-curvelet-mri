function y=getCollageImage(brain_vol, segmentationVolume, regionNumber, axis, padding)
    switch(axis)
        case 'axial'
            y=getCollageImageAxial_square(brain_vol, segmentationVolume, regionNumber, padding);
            %figure; % Visualizar mosaico
            %imshow(y, []);
        case 'sagital'
            y=getCollageImageSagital(brain_vol, segmentationVolume, regionNumber, padding);
        case 'coronal'
            y=getCollageImageFrontal(brain_vol, segmentationVolume, regionNumber, padding);
        case 'axial-transposed'
            y=getCollageImageAxialTransposed(brain_vol, segmentationVolume, regionNumber, padding);
    end
end