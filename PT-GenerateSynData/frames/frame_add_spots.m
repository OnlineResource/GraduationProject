function frame_t = frame_add_spots(frame_t,spots_t,spots_ini)

[H,W] = size(frame_t);
subframes = subframes_gen(spots_t,spots_ini);

for s = 1:length(subframes)
    h1 = subframes(s).upper_left_h;
    w1 = subframes(s).upper_left_w;
    h2 = subframes(s).upper_left_h + subframes(s).data_size_h - 1;
    w2 = subframes(s).upper_left_w + subframes(s).data_size_w - 1;
    add_data = subframes(s).data;
    if h1 < 1
        add_data(1:1 - h1,:) = [];
        h1 = 1;
    end
    if h2 > H
        add_data(end - (h2 - H - 1):end,:) = [];
        h2 = H;
    end
    if w1 < 1
        add_data(:,1:1 - w1) = [];
        w1 = 1;
    end
    if w2 > W
        add_data(:,end - (w2 - W - 1):end) = [];
        w2 = W;
    end
    frame_t(h1:h2,w1:w2) = frame_t(h1:h2,w1:w2) + add_data;
end

end