function C=fft3(A)
C=permute(fft(permute(fft(permute(fft(A),[2 3 1])),[2 3 1])),[2 3 1]);