function gray_imshow(I)
imshow(I,[min(I(:)) max(I(:))]);
end