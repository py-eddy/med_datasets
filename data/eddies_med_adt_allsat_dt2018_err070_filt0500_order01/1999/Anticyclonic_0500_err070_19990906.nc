CDF       
      obs    ,   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�p��
=q      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       QB�      �  \   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =�j      �     effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F��
=q     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �׮z�H    max       @vd�����     �  &�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           X  -|   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @���          �  -�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <49X   max       >�E�      �  .�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B/�      �  /4   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�y�   max       B/�=      �  /�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >QXg   max       C��m      �  0�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >2    max       C���      �  1D   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         P      �  1�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  2�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          Q      �  3T   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       Qb�      �  4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?��M:�      �  4�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >+      �  5d   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @F��
=q     �  6   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vd�����     �  <�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @O@           X  C�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @��           �  D,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  D�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��-V   max       ?���҈�     �  E�      $   0      @      %   "         M      '  O          <               0      2   $      J         2   "         /   G   	   
                     �N�Ow�PUN2w�Pi�N9WtOG��O���Nk��O��QB�N���P*|P�[6Nw�O���P n~OK\N�N��O�9O���N'��PM�iO�O�N2�LPc�O��`O{4O���O���O�F-NUuoO�wPJ�N�v�N/�)N�X^N\��O���N��DN�r�O �cO���o;�`B;�`B;�`B<49X<49X<T��<T��<u<u<�t�<���<��
<�1<�9X<���<���<�`B<�`B<�=+=+=+=��=��=8Q�=@�=P�`=T��=T��=Y�=e`B=e`B=m�h=m�h=�o=�C�=�O�=�\)=�\)=��-=��
=���=�j
 #/042/-'#�
#/<HU]_R?</#
�����������������������������������������W]t��������������g^W������������������#/<JTOH?</$"���������������������

���������#<Uanzzne^UH/����5BUZkh[B)�����bhkit~�������tphbbbb"&+;Ib{|}zsnbUI:%(',?Ng��������tgNB0(`anxz�{zvnea````````8OTWat��������tm[OB8������
�������������
���������������������������4)6BJECBB64444444444��}{|���������������������#<EH=/&
����)31*)ft��������������zvgf����)09>@>9)�������������������������BOPWh�����[B)�� ��)59AB?8685) ��������������������������)29961)�����������������������nv|��������������{un*26=<6*�������������������������)?DHB5)�����jcddimrz{���|znmjjjj)/6<6)bhht|����������~thbb('',/;<G?<5/((((((((����������������������������������������

#&/024/##

���������

������������	
�������(�5�A�E�N�P�N�N�A�5�.�(�����������������������������������������޹��Ϲܹ���������Ϲ�����������������������������������������������������������/�F�L�N�F�A�H�;�/�	����������������������������������������������������������`�m�y�������������y�m�`�T�Q�J�G�G�T�V�`�ûǻܻ������ܻʻ������u�{�������������T�a�m�n�o�m�i�a�X�T�N�H�T�T�T�T�T�T�T�T�����������������������������������{��������"��"�����Ǝ�C�*��
��*�\�pƳ�������(�/�(�%�&������� �������4�@�Y���������������Y�@�������'�4����)�B�M�T�N�B�6��������õóôø�����<�=�>�<�:�/�#��#�&�/�7�<�<�<�<�<�<�<�<�'�*�3�@�Y�f�h�e�Y�3�'���
���*�"�!�'���	��"�0�2�/�#�	���ʾ������������ʾ���/�<�H�U�W�a�c�i�j�a�U�H�A�<�2�/�,�#�/�/���(�.�4�A�B�A�4�(�������������������������������������������àìù����������������������ùõìæçà�)�B�N�[�`�a�`�d�s�g�[�N�C�5���������)�A�N�Z�\�b�Z�N�A�@�@�A�A�A�A�A�A�A�A�A�A�������(�7�B�F�A�6��������������Ŀۿ�y���������ÿÿ����������y�c�Z�W�U�`�f�y������������������������������������������4�S�k�f�Y�M�D�?�4���ѻ޻ڻ�����������пֿѿʿǿĿ��������y�m�g�b�j������"�/�;�H�T�a�h�m�u�m�a�T�H�0�/�!����z�������������������������������~�z�v�z���������˽˽νý����������y�o�j�g�q�y����4�A�I�O�Z�\�Z�W�M�A�4�(���������y���������������y�v�u�x�y�y�y�y�y�y�y�yĚĦĳĿ��������������ĿĳĦĚďċċčĚ��0�<�I�S�U�P�H�0�#�
�����������������ŇŔŠŭŹ��������ŹŭŠŔŐŇŇŇŇŇŇ�H�U�^�\�U�I�H�H�A�@�H�H�H�H�H�H�H�H�H�H�<�G�H�M�U�W�]�V�U�H�<�/�+�*�.�/�7�:�<�<Ŀ������������������ĿĹĿĿĿĿĿĿĿĿ�ֺ������������ֺ����������������ɺ������������������������|�s�g�s�{��������E�E�E�E�E�E�E�E�E�EuEiEdEiEiErEuE�E�E�E��#�/�<�H�N�U�`�a�d�a�Z�U�H�<�/�(�#���#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DsDrDwD{D� . K # c  @ $ J 6 Y S @ :  e L F F p d ' Z 8 Q  ' D : � D   ^ 9 $  Q Y Y K ? X _  !  �    z  q  �  M  �  �  �  �  	�  �    �  ^  a  �  Y  z    .  �  A  �  �  <  �  �  �  a  f  \  j  '  E    �  :  {  �  �  �  R  6<T��=0 �=]/<49X=��P<�C�=L��=<j<�t�=C�=\<�/=q��>�E�<�/=e`B=�{=�w<�=\)=�%=��
=#�
=�9X=��P=L��=�=��w=q��=��`=� �=��-=y�#=���>�=��=��-=�Q�=�{=��=���=��=�S�>{�mB�B��B�B!�BqBeB>�B"�EB��BԵBh$B~�B'B	(RB��Bw|B"<B��B# �B�BAPBíBZB2B�B��BE�BH�B��BP�B,$lBy,B/�B]�BvA��B�B�B\=BBPB�!B�gBB��B�vB�wB�nB!k�B
ĒB��B@B"BVSB��B�2B��B'@\B	?B�1B�B?�B�fB#fB+B?]B>�B<)B=�B�MB��B�>B;�B�qB��B,?�B�}B/�=BM.BGUA�y�B4�BSB@ B�WB��B>�B<SB��A�*A���>QXg@Z�A���AH�*Ak�@�>�A�#)A�BJ�A3��@�@xA�rIAe?�GAU��A���A6��@XA�m|A�)�A��aA��Ap�A��!@�s�As�6A���A��bAs�A7{An��AᙫA�! A��AĦyA��>A��m@8�A�\�C��mA���C��'A�XA�u�>2 @& A��LAH��Ak3@�)�A�p�A��B�ZA4Ҷ@�#A��A#?ǮEAW6�A�V�A7v�@[�A�&�A�u�A�@JA�eAquA���@��{As��A��GA���AHIA9tAoA�~?A�x�A���AĈpA���A�~�@;�(A�t�C���AÏ�C��      %   0      @      &   "         N   	   (  P          <               0      3   %      J         3   "         /   H   
   
                      �         '      3         %      !   Q      -   1      !   )               %      3         5         %      !         +               !                     %      )                  Q      )   !         %               %      +         /         %      !         )               !            N�N��ZO��N2w�P<�N9WtOJO8��Nk��Oh��Qb�N���P�oOњNw�N�ըP�!O�TN�N��OFāO���N'��P�)O�2N2�LPP`O/�.N�_O���O7E}O�F-NUuoO?��P:`EN�v�N/�)N��N@�O���N��DN�r�O �cO72  :  �    Q  �  �  &  �  �  �  �  C    $  o  �  9  �  �  �  �  P  �  �  �  #  �  �  �  �  �  4  &  v  �  P    \  �  �  �  �  �  ��o<ě�<D��;�`B<��
<49X<�1<ě�<u<�t�<���<���<���>+<�9X=�w<��<�h<�`B<�=#�
=+=+=8Q�=]/=8Q�=P�`=y�#=]/=T��=�+=e`B=e`B=�O�=}�=�o=�C�=��=�hs=�\)=��-=��
=���>	7L
 #/042/-'#"#'/3<CHPLH></)#""""����������������������������������������b[^gt�������������tb������������������ #$/<AHGD<8/#    ���������� ������������

���������"#/<Uagd`[UH</#"����4BQ^kg[B)�����bhkit~�������tphbbbb#*/CIeswxvpnbUI@.'(#>;;@N[gt������tg[NE>`anxz�{zvnea````````dbehmt���������ythdd�������

������������	����������������������������4)6BJECBB64444444444��������������������������#<EH=/&
����)31*)������������������ )+241*)�����������������������6BSh�����h[B)�)57:750)	��������������������������)29961)�����������������������nv|��������������{un*26=<6*�������������������������)>CFB5)����jcddimrz{���|znmjjjj)/6<6)ttu������������ttttt*'(-/9<D=<4/********����������������������������������������

#&/024/##

���������

����������	

��������(�5�A�E�N�P�N�N�A�5�.�(������������������ ������������������������ùϹܹ���� ����Ϲ������������������ú�������������������������������������������"�2�=�D�F�8�/�"�	�����������������𾌾��������������������������������������T�`�m�y�����������y�m�`�V�T�P�Q�T�T�T�T�����ûŻл׻Իлû����������������������T�a�m�n�o�m�i�a�X�T�N�H�T�T�T�T�T�T�T�T����������������������������������������������!������u�C�)���	�*�]�rƳ�������(�/�(�%�&������� �������Y�������������f�Y�@�'�����'�4�M�Y�������.�6�8�6�+���������������������<�=�>�<�:�/�#��#�&�/�7�<�<�<�<�<�<�<�<�3�@�L�P�Y�^�`�Z�Y�L�@�>�3�'�#�'�(�/�3�3����	��"�-�0�+��	���ʾ������������о��<�H�U�V�a�c�h�h�a�U�H�C�<�4�0�3�<�<�<�<���(�.�4�A�B�A�4�(�������������������������������������������ù������������������������ýùóíñ÷ù�)�B�N�[�`�a�`�d�s�g�[�N�C�5���������)�A�N�Z�\�b�Z�N�A�@�@�A�A�A�A�A�A�A�A�A�A������2�=�B�A�5����ѿ������������ݿ꿆�����������������������y�x�t�x�y�������������������������������������������������4�P�`�e�M�A�;�4���ۻڻ�ܻ�������������ĿĿ����������������}�����������/�;�H�I�T�a�e�m�n�m�a�T�N�H�;�4�/�*�/�/�z�������������������������������~�z�v�z�������������������������~�y�t�q�u�y�~����4�A�I�O�Z�\�Z�W�M�A�4�(���������y���������������y�v�u�x�y�y�y�y�y�y�y�yĦĳĿ������������ĿĳĦĤĚĔđĒĚĦĦ��0�<�I�Q�S�N�F�0�#�
���������������
�ŇŔŠŭŹ��������ŹŭŠŔŐŇŇŇŇŇŇ�H�U�^�\�U�I�H�H�A�@�H�H�H�H�H�H�H�H�H�H�<�=�H�S�U�Y�U�Q�H�<�/�-�,�/�2�<�<�<�<�<Ŀ������������������ĿĺĿĿĿĿĿĿĿĿ�ֺ������������ֺ����������������ɺ������������������������|�s�g�s�{��������E�E�E�E�E�E�E�E�E�EuEiEdEiEiErEuE�E�E�E��#�/�<�H�N�U�`�a�d�a�Z�U�H�<�/�(�#���#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D{D�D�D�D� . 4 ! c  @  & 6 B R @ 1 
 e % > = p d  Z 8 X  ' C  } D & ^ 9    Q Y : K ? X _    �  �  !  q    M  !  �  �    	�  �  U  �  ^  �  �  ,  z    �  �  A    '  <  |  p    a  �  \  j  �      �  �  f  �  �  �  R  |  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  :  4  /  (        �  �  �  �    X  2  
  �  �  �  �  p  2  y  �    ;  U  l  �  �  �  �  n  :  �  �  5  �  7  �  l  �        �  �  �  �  a  &  �  �  g     �  �  L  �  u  �  Q  F  <  2  '        �  �  �  �  �  �  �  �  �  x  j  [  J  �  �  �  �  �  �  l  B    �  �  A    �  �  B  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  R  9  %    �  
    !  $  &  $       	  �  �  �  �  N    �  �    4  %  �    $  Q  u  �  �  �  �  o  L    �  �  e    �  G  �  6  �  �  �  �  �  w  o  c  V  I  <  0  #       �   �   �   �   �  o  |  �  �  �  �  �  �  o  I    �  �  �  m  C    �  @  
  �  �  c  2  '  N  V  Q  B  *    �  {  1  �  L  �    a  �  C  >  9  6  3  2  4  6  9  =  ?  B  C  C  C  D  E  D  ?  9  �  
          �  �  �  �  �  j  :     �  �  �  x  �  y  �  I  �  V  m  6  �  �  "    �  )  m  j    N  �  �    �  o  w    �  �  �    x  n  ^  M  =  (    �  �  �  �  S  #  �  N  \  g  n  r  v  ~  �  �  y  a  =    �  �  I  �  _  �  -  !  .  !        �  �  �  �  `    �  v  "  �    X  �  �  �  �  �  �  �  s  [  B  &    �  �  �  i  7    �  {    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  1  �  �  )  �  )  �  N  P  %  �    ?  <  )  
  �  �  �  \  #  �  �  ]  �  4     �  �  �  �  �  �    v  l  b  U  H  :  *    
  �  �  �  �  �  �  �  �  �  �  �  W  8    �  �  �  �  �  K  �  �  �  �  B    .  K  t  �  �  �  �  �  �  �  �  x  ?    �  O  �  �  !  #    
  �  �  �  �  �  �  �  w  ]  E  /       �   �   �   �  �  �  �  �  �  �  d  w  �  f  )  �  �  �  p  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  C    �  s    �  +  �    i  _  w  �  �  �    j  T  <    �  �  �  c  %  �  �  �  �  f  L  1    �  �  �  �  a  $  �  �  ]  �  ~  9  e  Q  |  �  �  �  �  �  �  �  �  �  �  �  {  5  �  v    z  �  5  4  �  �  �  �  �  �  t  N  $  �  �  d  !  �  �  �  �  �    &        �  �  �  �  �  �  �  p  W  >  $  
   �   �   �   �  �  1  S  i  u  u  h  S  8    �  �  F  �  �    �  �  �  �  �  �  �  �  �  �  `  /  �  �  y  D  �  �  I  �  s  �  �    P  5    �  �  �  �  {  X  6    �  �  �  �  �  �  �  �  �        
  �  �  �  �  �  x  ^  B  %    �  �  �  �  g  K  �    &  I  [  V  J  6    �  �  �  n  ,  �  �  \    �  J  �  �  �  �  �  �  �  c  ?    �  �  �  �  �  e  L  f  �  �  �  �  �  �  i  I  '    �  �  �  M    �  m    �  X  �  .  �  �  �  �  r  ^  I  1    �  �  �  �  �  �  o  Y  I  9  )  �  �  �  d  -  �    �  �  B  �  �  x  7  �  �  Y     �  ,  �  �  �  h  >    �  �  s  :    �    1  �  w  �  b  �  �  �  �  "  N  p  �  t  >  �  �  2  �    ,    �  �  �  �  �