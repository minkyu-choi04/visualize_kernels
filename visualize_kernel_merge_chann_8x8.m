kernel = conv1_k;

idx_chan = 3;
sz = size(kernel);
num_row = 8;

imax = max(max(max(max(kernel))))*0.2;
slide = ones(sz(3), 5)*imax;
for c=1:3
    idx_chan = c;
    for idx_row=1:num_row 
        for i=cast((sz(1)/num_row), 'int32')*(idx_row-1)+1 : cast((sz(1)/num_row), 'int32')*(idx_row)
            if i==cast((sz(1)/num_row), 'int32')*(idx_row-1)+1
                hss1 = squeeze(kernel(i, idx_chan, :, :));
            else
                hss1 = cat(2, hss1, slide, squeeze(kernel(i, idx_chan, :, :)));
            end
        end

        %idx_row, i
        if idx_row==1
            hss = hss1;
        else
            slide_h = ones(5, size(hss1,2))*imax;
            hss = cat(1, hss, slide_h, hss1);
        end
        
    end
    
    if c==1
        out = squeeze(hss);
        %out = unsqueze(out, 1);
    else
        out = cat(3, out, hss);
    end
end


figure(26)
%imagesc(out);
%colorbar;
%axis image

imin = min(min(min(out)))
imax = max(max(max(out)))
imean = (imin+imax/2);
out_n = (out - imin) / (imax-imin);
min(min(min(out_n)))
max(max(max(out_n)))

imagesc(double(out_n))
colorbar;
axis image


