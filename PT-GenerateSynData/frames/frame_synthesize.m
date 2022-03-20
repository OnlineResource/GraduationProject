function frame = frame_synthesize(background,spots,spots_ini)
[~,~,T] = size(background);
frame = background;
for t = 1:T
    frame(:,:,t) = frame_add_spots(frame(:,:,t),spots(t),spots_ini);
end
frame = uint16(frame);
end