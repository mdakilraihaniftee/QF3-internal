% Generic driver: reads R datasets of length N (raw float64 binary, row-major) and calls
% ci(X,k) for k=1..10 on each; writes R x 20 matrix [a1 b1 ... a10 b10] as raw float64.
% Only the input/output behaviour of ci() is used.
function run_batch(infile, outfile, seed, R, N)
  rand('twister', seed);
  fid = fopen(infile,'rb'); D = fread(fid,[N R],'double')'; fclose(fid);
  out = nan(R,20);
  for r = 1:R
    x = D(r,:);
    for k = 1:10
      try
        [a b] = ci(x,k);
        out(r,2*k-1) = a; out(r,2*k) = b;
      catch
      end
    end
  end
  fid = fopen(outfile,'wb'); fwrite(fid,out','double'); fclose(fid);
end
