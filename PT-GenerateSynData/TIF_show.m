addpath(genpath(pwd))
% im = imreadGrayTiff('200502 105 93gfpsec61btf lyso647 exp200ms in2s 057 crop1.tif');
% T = size(im,3);
% for t = 2:2:T
%     pause(0.1)
%     gray_imshow(im(:,:,t));
% end

im = imreadGrayTiff('eval_00.tif');
T = size(im,3);
for t = 1:T
    pause(0.1)
    gray_imshow(im(:,:,t));
end

