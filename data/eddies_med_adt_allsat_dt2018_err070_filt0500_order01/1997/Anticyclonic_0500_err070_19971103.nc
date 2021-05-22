CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�t�j~��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nn�   max       P���      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =ě�      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(�   max       @E��
=p�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vp�����     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @M@           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��           �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >N�      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       B �   max       B,@-      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       B @�   max       B,?�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�c�      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�*�   max       C�e�      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          O      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nn�   max       P�Lq      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?�@��4m�      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       =�x�      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(��   max       @E�p��
>     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vo�
=p�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         DA   max         DA      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?�?�     �  K�         '   	            
   7   (            $      '         q   
                     ;   0         {   �   !      *   !                        #            L         N�?�N��O��bN21�O+gLOWz�N�ՂOq3#P���O��!N�pN��KO�FyP��NB	O��Ng�N�gOP0 N2�7N�}�O,P9O$�O�w�N�l�N�IoP�LqP�	O���N��O���PRxO� N�i�O���O��8N�o�Nz��Ng��N�PXN24?Nn�N�jO m�NH(-O 4PN���O��2OήN6K�N�l�����#�
�ě��o;�o;�`B;�`B<u<�o<�t�<�t�<���<�j<���<�/<�`B<�h<�<��<��=o=\)=t�=�P=��=��=�w=�w='�=<j=@�=L��=L��=L��=P�`=T��=]/=aG�=aG�=e`B=e`B=ix�=m�h=y�#=�7L=�\)=���=���=���=�1=ě�smjlsst���������vtss^ZYanzz��zwnea^^^^^^ #/<HUanu}miaUH;/' INOU[ghjg`[NIIIIIIII
	#0<ADD><90##
=DDN[gt|~}yvvtg[TNH=������������������������)5;:<<85����5EMW``X)�������@=>AJ[ht�����tph[OF@��������������������/168BOQVOLDB?6//////gfgcfhnz���������zmg]ah���������������m]oyz��������zoooooooo��������!�����\ht��������tih\\\\\\��������������������������)5=AB>5)����2355BCIIB52222222222��������������������468;AHUalnpnndaUNH<4���������������������������� ��������`_hht������tkh``````��������������������5Nt�����������dW��������
'142&�����������������������������������������
!"�����������NbkiTG5)������� 
#*,'#
���tyz������������ztttt���6BLSUTLB)�����������������������)+)%�����

#%--'#�����



��������
#&#
RITUZbnpnebURRRRRRRR����������������������������

�����{|�����������������#%/<GGGB<:/##��������������������niinz������������vqn��)16:63+)���
)3/)









������

�������a�n�zÇÓàçäàÚÓÇ�z�n�i�d�a�a�a�aE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�����������
����������������������b�j�n�{Ł�|�{�n�h�b�_�^�b�b�b�b�b�b�b�b�x���������������������x�r�l�\�^�_�l�p�x�`�m���������������y�m�`�T�P�Q�X�T�I�T�`�����������������������������������������������Ŀѿڿݿ�ܿԿ׿ѿĿ�������������Ƨ�������"�'�����ƧƎƁ�R�6�!�*�C�\�uƧ�4�A�M�Z�f�u�{�{�}�s�f�A�5�!�����(�4�Ϲܹ���������������ܹ׹Ϲι˹ϹϹϹϼʼּ������ּμʼ��������ʼʼʼʼʼ�ŔŠŭŹ��������������������ŭšŔŐőŔ�A�M�Z�t������������s�f�Z�A�1�%�#�&�4�AĿ����������������ĿľļĿĿĿĿĿĿĿĿ����������¼˼ʼ���������r�f�Z�_�_�r������»����������������������������������a�d�n�y�u�n�n�h�a�\�\�\�V�a�a�a�a�a�a�a��0�<�O�V�U�W�O�C�0�
����Ŀ���������
�ĚĦĳļĶĳĦĚĐĕĚĚĚĚĚĚĚĚĚĚÇÓàçìùüùöìàÓÇ�z�y�z�|ÅÇÇ�a�m�z�����������������z�y�m�k�c�a�_�\�a���	��"�&�%�"� ��	����������������������������������������������z�x�����-�:�F�F�Q�K�F�:�-�!���!�$�-�-�-�-�-�-����!�-�.�:�<�:�-�!�����������������������������s�A�,���ڿ���Z�������������"�/�;�H�O�X�Y�T�/�"��	������������#�(�)�)�.�$���
���������������
���)�6�B�B�G�F�B�<�6�/�)�(�%�&�)�)�)�)�)�)D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D}D�D�D�D����6�D�M�J�D�%��������������������.�G�T�`�m�p�i�G�.�"����׾ξؾ����.�
������#�&�#��
�� �����
�
�
�
�ܻ�����������ܻлŻ����������ǻ������������������������������������������M�Z�f�i�l�l�h�f�]�Z�R�M�F�F�G�H�M�M�M�M�4�A�M�Z�e�f�r�f�Z�M�A�8�4�/�4�4�4�4�4�4�������������������������������������������������������������������������������ĦĳĿ��ĿĻĵĳįĦğĠĦĦĦĦĦĦĦĦ�����	���������������������y��������������������y�u�x�y�y�y�y�y�yE�E�E�E�E�E�E�E�E�E�E�E�E�E�EvElEpEuE|E��g�s�������������s�j�g�b�g�g�g�g�g�g�g�g�B�N�[�b�g�q�s�g�[�N�B�6�5�/�5�6�B�B�B�B�������������������������������������������ɺֺۺ׺Ժκ˺ɺ����������{�{�~���������ƺȺźúĺ����������������������������t�x�t�p�l�n�t�t�t�t�t�t�t�t�t�t�*�6�C�G�O�R�Z�\�O�C�:�6�/�*�&�$�*�*�*�* 7 / C S 0 2 U $ > - 9 5 , ; r $ u m @ N , 9 ( A ' m f = 2 V / A T u < S R Z 3 P A [ ( N 8 3 R Q j n S    3  �  <  k  h  �  �  �  �    �  �  �  �  �  �  �  �  9  J  �  z  `  �  �  �  �  n  -  �  D  �  \  J  �  M  ?  �  m  �  V  E  �  v  f    �  }  g  �  	�ě�;ě�=+;�`B<�o<�1<D��<���=�hs=m�h<���<�=P�`=u=o=�7L=#�
=�P>t�='�=D��=}�=H�9=T��=8Q�=,1=Ƨ�=�-=��=�o>-V>N�=���=m�h=�v�=�{=�+=y�#=y�#=���=y�#=}�=�hs=ě�=�hs=ě�=�->�R=\=�^5=�S�B
"B�mB�B�	B%woB	0B"��B.!B�B��B ��B.�B hiB��B �B#0^B=B�HB�*B��B!��B��B"��B�BK�B�YB	��BS;B��BpfB;�B?�B�qBA�BRBWB��B$s�B%gBM�B xB'�B,@-B��B�Bl�BM�B��B|�B@Bo�B
@5B��B�B��B%�,B	JB#4CB;B¨B��B ��B?gB O�B��B @�B#?�BE�B>B>MB��B"CNB��B"��B��B%B�uBBB?:B��BEB?jB@B@BA�B�%BN�B�(B$V	B%:�B?�B�B'�zB,?�B��B�]B=FB�tB��B�BG�BBuA��C�c�A�m�A�o�@�VEAl�@��{Av�yB	�A=@	>��A �JA��aA?��A�=.@��@��A�f3A�W�A�o�A���A�AZd�A���@x��@j�_A��#A�-A� xA�^IC��A��GA^�A��7@��pA��A>t�A=AI��A�GA�`�A��A��C��A���A���A��j@ 8�@&\eA��mB ��A���C�e�A�~fA��C@��`Am�@�&kAv��B��A=1A>�*�A � A�}�AA&�A�p@��@���A���A�|�A�y�A�_�A�RAZ��A�Z�@v�s@c�DA�b�A�\�A��:A׀�C���AӁ�A`�A�}�@�m�A���A? A= AIfA�A�nA�A��C�fA�L�A��A���@$}@#�(A���B J�         (   
               8   )            $      (         r                        ;   1         {   �   "   	   +   "                        $            M                                    E               '      !         -               !         O   %            /   '      !                                       %                                    ?               !                                       O   #               '                                                      N��N�7jN���N21�OY?N@��N�ՂO/�BP�>N�[�NU�wN��KO�<\O�kNB	Oj�VNg�NM2�O���N2�7N�}�O��N�T�Oy�@N�l�N�IoP�LqO���OS&�Nh1O�pO���O� N�i�O���O��8N�o�Nz��Ng��N�PXN24?Nn�NO��Oi�NH(-N���N���O�%OήN6K�N�l  C  +  ~  7  �    �  �  	    Y  %  U  �  .  ,  -  �  |  A    �  �  �  �  �  a  �  u  w  �  �  �    i  i  �  �  8  p  �  t  �  
  n  �  {  
�  8  l  ϼ�j��`B<T���o;ě�<u;�`B<�C�<�1=�P<���<���<���=+<�/=��<�h<��=��T<��=o=�w=�w=�w=��=��=�w=49X=8Q�=H�9=�^5=�x�=L��=L��=q��=T��=]/=aG�=aG�=e`B=e`B=ix�=}�=�o=�7L=��-=���=��=���=�1=ě�olntv{�����������too`]]ansz~ztnha``````.,//<HOUWUPH</......INOU[ghjg`[NIIIIIIII!#0<?BB<<50#XT[_gtuwtjg[XXXXXXXX������������������������#)59:5)	������5BIR_]J)�����KKOOZ[htwyxttjh[VOKK��������������������/168BOQVOLDB?6//////klhlrz����������zumkmr����������������umoyz��������zoooooooo����������
�����\ht��������tih\\\\\\�������������������������)-36750)�2355BCIIB52222222222��������������������;;<=CHPU[aimka^UH<;;����������������������������������������`_hht������tkh``````��������������������5Nt�����������dW�������
#)-/- ���������	��������������������������������

���������+9BFEA8)������� 
#*,'#
���tyz������������ztttt��)6BFNQOGB)����������������������)+)%�����

#%--'#�����



��������
#&#
RITUZbnpnebURRRRRRRR��������������������������

������{|�����������������"##*/<@BC></##""""""��������������������z{���������������~{z��)16:63+)���
)3/)









������

�������n�zÇÑÓàáàÔÓÊÇÆ�z�n�n�f�e�n�nE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���������	����������������������������b�j�n�{Ł�|�{�n�h�b�_�^�b�b�b�b�b�b�b�b�x���������������������y�x�l�b�d�l�t�x�x�m�y�����������y�r�m�k�k�m�m�m�m�m�m�m�m���������������������������������������������ĿѿտοѿտѿϿɿ�����������������Ƨ����������!�����ƚƎ�u�V�H�=�\�uƎƧ�A�M�Y�Z�f�h�n�i�f�Z�M�A�A�4�1�1�4�9�A�A�Ϲܹ����������ݹܹϹϹ̹ϹϹϹϹϹϼʼּ������ּμʼ��������ʼʼʼʼʼ�ŠŭŹ����������������ŹŭŤŜŖœŔřŠ�M�Z�k�}����������s�f�Z�C�6�1�.�1�8�A�MĿ����������������ĿľļĿĿĿĿĿĿĿĿ�r��������������ļ���������y�q�i�i�k�r�����»����������������������������������a�n�x�s�n�k�c�a�`�^�_�W�a�a�a�a�a�a�a�a�
��#�0�3�<�@�=�6�0�#��
�������������
ĚĦĳļĶĳĦĚĐĕĚĚĚĚĚĚĚĚĚĚÇÓàçìùüùöìàÓÇ�z�y�z�|ÅÇÇ�m�z�}�������������������z�o�m�h�d�j�m�m����	��!� ���	�����������������������������������������������|�����-�:�F�F�Q�K�F�:�-�!���!�$�-�-�-�-�-�-����!�-�.�:�<�:�-�!�����������������������������s�A�,���ڿ���Z����������"�/�;�H�K�T�T�H�;�/�"��	�����������
��!�$�*����
���������������������
�)�6�>�B�E�D�B�8�6�3�+�)�'�)�)�)�)�)�)�)D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����)�2�4�2�)����������������������.�G�T�`�m�p�i�G�.�"����׾ξؾ����.�
������#�&�#��
�� �����
�
�
�
�ܻ��������������ܻл˻��������л������������������������������������������M�Z�f�i�l�l�h�f�]�Z�R�M�F�F�G�H�M�M�M�M�4�A�M�Z�e�f�r�f�Z�M�A�8�4�/�4�4�4�4�4�4�������������������������������������������������������������������������������ĦĳĿ��ĿĻĵĳįĦğĠĦĦĦĦĦĦĦĦ�����	��������������������ｅ�������������������y�w�y�~������������E�E�E�E�E�E�E�E�E�E�E�E�ExEuEoEtEuE�E�E��g�s�������������s�j�g�b�g�g�g�g�g�g�g�g�N�U�[�g�m�l�g�[�N�B�>�7�B�G�N�N�N�N�N�N���������������������������������������������ɺʺʺǺźú��������������������������ƺȺźúĺ����������������������������t�x�t�p�l�n�t�t�t�t�t�t�t�t�t�t�*�6�C�G�O�R�Z�\�O�C�:�6�/�*�&�$�*�*�*�* @ 0   S , C U 1 6 % N 5 + 3 r ( u ~ ' N , + ! 1 ' m f 2 % R   * T u A S R Z 3 P A [  A 8 4 R @ j n S    �  |  �  k  (  b  �  s    
  �  �  6  �  �  �  �  �    J  �  $  �  �  �  �  �  �  �  �  L  q  \  J  u  M  ?  �  m  �  V  E  Z  ?  f  �  �    g  �  	  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  DA  �    /  C  @  4  #    �  �  �  �  �  �  c  <    �  �  y      "  (  +  )  !      �  �  �  y  X  5    �  �  �  �  4  |  �  �  �  �  #  V  t  ~  v  [  /  �  �  Z  �  n  �  ~  7  ,     $  *      �  �  �  Y  (  �  �  �  E  	  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  {  [  8    �  �  �  a  z  �  �  �  �  �  �  �  �  �       �  �  �  �  ;  �  v  �  �  �  �  ~  v  n  f  ^  V  N  G  ?  8  0  )  /  8  A  J  �  �  �  �  �  �  �  �  �  �  k  K  (    �  �  �  q  <    �      �  �  �  �  �  �  c     �  V  �  �  S    �     5  �  �  �  �  �                �  �  �  c    �  Q  "  :  E  Q  \  f  p  s  o  l  d  [  Q  D  6  )           �  %        �  �  �  �  �  �  �  �  v  e  V  J  =  0    �  H  R  U  O  ?  &    �  �  �  Y  &  �  �  �  `    �  �     g  �  �  �  �  �  �  �  }  j  V  =    �  �  g    �    6  .  $             �  �  �  �  �  �  o  T  :  �  �  u  1  �  �  �    &  ,  $    �  �    �  �  �  n    �  �  9  �  -  j  �  �  �  �  �  �  �  �  �  �  �  �  �  w  <  �  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  -  �  �  �    .  L  g  y  u  >  �  v  
�  
E  	g  T  �    �  A  M  V  @  +    �  �  �  �  �  �  �  ~  w  r  z  �  �  �    �  �  �  �  y  c  :    �  �  �  �  �  �  \  &  �  �  K  �  �  �  �  �  �  �  �  �  n  T  1    �  �    �  �    K  �  �  �  �  �  �  �  �  �  �  �  �  |  d  H  (    �  n   �  U  W  p  �  |  u  f  U  G  <  '    �  �  n  '  �  �  N   �  �  �  �  �  v  i  ]  Q  E  ;  0  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    t  h  \  O  C  a  P  -  �       
  �  �  �  �  �  �  p  #  �  j    �  K  w  �  �  �  �  �  z  i  T  :    �  �  l    �  \  �  �  O  3  D  k  u  q  h  X  A  #  �  �  �  m    �  _  $  �    �  f  r  t  w  s  i  P  '  �  �  M  �  �    �    �  �  [   �    �    a  �  �  �  �  �  E  �  4  �  �  �  f  �  
�  �  :  �  d    �  �  L  �  �  �  �  g  �  }  �    �  
�  �  e  �  �  r  R  )  �  �    1  ;  4    �  �  �  ]    �    �          �  �  �  �  �  �  �  �  �  �  �  �  �  C  ~  �  �    >  X  f  h  \  H  +    �  �  k    �  7  �  �  i  �  �  i  V  G  =  .    �  �  �  w  G    �  �  q     �  $    �  �  �  �  �  �  �  �  �  �  ~  ^  8  �  �  L  �  �  T   �   �  �  �  �  �  �  �  �  �  �  {  q  g  \  P  ?  -          8  /  %            �  �  �  �  �  �  �  �  �  �  x  g  p  X  6  	  �  �  N    �  [  �  �  6  �  u    �  J   �   v  �  �  �  �  �  p  ]  J  9  *      �  �  �  �  �  �  �  l  t  j  `  V  K  >  1  $    �  �  �  �  �  �  v  Y  ;       M  b  t  �  �  �  �  �  y  j  S  2    �  �  �  F  �  C  �  
  
  
  
  	�  	�  	�  	Z  	  �  T  �  a  �  E  �  �  c  �  �  n  g  `  Y  S  I  8  '      �  �  �  �  �  k  Q  8      �  �  �  �  �  �  k  R  6    �  �  �  M    �  �  D  �  �  {  k  W  A  *    �  �  �  �  �  Y  0    �  �  �  x  g  W  	�  	�  
@  
b  
{  
�  
�  
y  
X  
*  	�  	�  	?  �  .  �  �  �  �  �  8    �  �  �  h  6    �  �  �  �  �  P  �  �  1  �  �  8  l  J  (    �  �  �  �  _  >    �  �  �  �  F  �  �  �  V  �  �  �  �  k  P  *    �  �  �  _  0  �  �  v  �    A  �