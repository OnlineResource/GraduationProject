function subframes = subframes_gen(spots_t,spots_ini)
S = length(spots_t.ID);
subframes = struct([]);
stretch_ratio = 1 + 0.2*(spots_t.velocity > spots_ini.velocity_h_threshold);
rotate_degree = -spots_t.direction/pi*180;
for s = 1:S
    diameter = spots_t.size(s);
    data_size = round(2*diameter);
    data_center_h = round(data_size/2);
    data_center_w = data_center_h;
    h = repmat(1:data_size,data_size,1);
    w = repmat(1:data_size,data_size,1).';
    D = abs(((h - data_center_h).^2+(w - data_center_w).^2).^0.5);
    D_mask = D < D(round(data_size/2),round(data_size/4));
    data = spots_t.intensity(s) - spots_t.intensity(s)*(D/max(max(D.*D_mask)));
    data(data < data(round(data_size/2),round(data_size/4))) = 0;
    data = uint16(abs(data));
    
    % 拉伸旋转处理
    data = imresize(data,round(data_size*[1/stretch_ratio(s) stretch_ratio(s)]));
    data = imrotate(data,rotate_degree(s),'bilinear');
    
    subframes(s).data_size_h = size(data,1);
    subframes(s).data_size_w = size(data,2);
    subframes(s).upper_left_h = round(spots_t.h_position(s) ...
        - subframes(s).data_size_h/2);
    subframes(s).upper_left_w = round(spots_t.w_position(s) ...
        - subframes(s).data_size_w/2);
    subframes(s).data = data;
end

end