function background = background_gen(size_frame,background_range)
[H,W,T] = deal(size_frame(1),size_frame(2),size_frame(3));
background = randi(background_range,H,W,T);

% 对背景进行均值滤波
f_mean = fspecial('average', 3);

background = 0.25*background + 0.75*imfilter(background,f_mean,'symmetric');

% frame = 0.25*frame + 0.75*imgaussfilt(frame);

background = uint16(background);
end
