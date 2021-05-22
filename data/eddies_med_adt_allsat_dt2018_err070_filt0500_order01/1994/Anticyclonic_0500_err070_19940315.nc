CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�?|�hs      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       PYξ      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       =��
      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�33333     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ҏ\(��    max       @vR�Q�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @N            h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�`          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��9X   max       >         �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��[   max       B,��      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�`      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >l�   max       C�}      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�`�   max       C�z0      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          #      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       O�3�      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��+J   max       ?���C�\�      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =��      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E���R     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vR�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�j           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?   max         ?      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?���C�\�     �  K�      
      3      $      	                                       .                     	      ,                            
      S   
               &   *      0   N��N��}O�3�O��Nd��O�J�N�.(N�p�NwJ�Nc�N;�O��NԔN�)O��PYξOm��N��wNH��N��P�QO�ŒO��N�k8N��{OӰO��N��N�LO��JNF)�O�ʡO��,N ��N��N&?Ns��O�6N���N�+-Ot1VN���O�N��Op�/O���O�V�O���NBi�O�u�N܈V��h������t���t��D����o;o;D��;��
<o<o<o<o<t�<t�<#�
<49X<49X<D��<e`B<u<u<�o<�t�<���<��
<��
<�1<�9X<�j<ě�<���<���<�`B<�`B<�h<�=o=�w=,1=,1=,1=D��=T��=Y�=]/=aG�=e`B=y�#=�t�=��
!!#0<>?=<:0#!!!!!!IFNO[fgrtutog^[NIIIIXSWV]gt����������t`Xnov��������������xnchktv�����thcccccccc���#/<FSTRH</#������������������������Z[^hqt{}tmhc^^[ZZZZ)5;=5,)(��������������������##(/5;<<<</)/6/&#�������������������������������������������������������������������!(("�����������
&!&G>/����������������������������������������������������������������EEO[ehttvth[YOEEEEEE "/;ABGJCC;"
gdels|�����������hg��������������������#!#'0<IUJIA<10#####)5BLLB=50)%�����������

������ �
#/@D;3,,&
����������	����������������������������������������������(!)-5BBB@?5)((((((((��������������������!)5:@EFC?5RUajntunjaUURRRRRRRR�����������������������������������������������������������,((+/6<DHRUZXUPH</,,A779BO[a[ZOMBBAAAAAA;5311;BHOTXUTH=;;;;;��������

����������������������������������
���UOIH><:<<HUUUUUUUUUU����#,16) �����)6BOW]VKC:6)������������������������������likmtz~�zsmllllllll�������������������������������������������������������޽�����������������������������������������������Óì������������������ùìàÓ�z�q�nÇÓ�O�[�h�r�xĀ�}Ćā�h�[�O�6�)����)�6�O�L�O�Y�d�e�l�e�Y�L�F�C�I�L�L�L�L�L�L�L�L�y�����������y�m�T�G�:�5�3�4�;�G�X�`�r�y�n�{łŇŉŇ�}�{�n�b�_�a�b�j�n�n�n�n�n�n�ʼмʼƼ����������������������ǼʼʼʼʾZ�f�i�l�n�f�Z�P�M�L�M�S�Z�Z�Z�Z�Z�Z�Z�Z�ĿǿɿȿĿ��������ÿĿĿĿĿĿĿĿĿĿ������������������������������������������Ϲܹ����������Ϲù��������������������/�;�H�S�T�Y�X�T�K�H�;�2�/�$�'�/�/�/�/�/�����������������������������������������O�\�uƁƏƚƨưƩƎ�u�h�\�O�G�<�:�F�G�O����;�T�a�h�{�{�a�H�;�������������������!����������������������������àâìùü��úùìàÓÎÓÓÖÝàààà����(�)�(������	���������ݿ�����ݿܿѿϿϿпѿ׿ݿݿݿݿݿ��/�;�T�a�o�u�w�s�a�H�;�/��	���������	�/�M�Z�f�����������������s�f�M�A�9�A�K�M��������!�"�-�0�-�!������������f�r���������������}�r�m�f�e�e�f�f�f�f���������������������������������򾱾��ʾ׾۾�۾׾˾ʾ��������������������(�A�M�\�u������������s�Z�M�4�(����(�`�m�s�y���y�w�m�g�`�T�K�I�O�T�W�`�`�`�`�4�A�M�U�Z�b�Z�P�M�A�=�4�-�(�&�'�(�*�4�4�5�B�[�a�i�k�i�[�B�)���������������5�;�G�P�T�Z�X�T�Q�G�;�7�:�;�;�;�;�;�;�;�;����������н��������
���齷��������)�3�B�N�[�j�t�}�y�t�g�N�B�5�)�����N�N�S�N�J�A�5�3�5�8�A�J�N�N�N�N�N�N�N�N�<�H�J�U�[�U�S�H�@�<�/�$�#�!�#�&�/�;�<�<���ʼʼռμʼ������������������������������������ĽŽĽ��������������������������������������� �����������������޻лܻ�������������߻ܻллллллл�����(�5�>�5�/�(�"����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��l�y�����������y�o�l�k�d�l�l�l�l�l�l�l�l���ʾ׾�����"�,�*��	�����׾ξʾ���E*EEEEEEE*E*E5E*E*E*E*E*E*E*E*E*E*�#�<�I�M�X�\�[�U�I�<�0�+�&� ��
��
��#�e�~�����������ʺź����������r�i�[�T�_�e���������������������������|�t�v�}�������@�L�J�D�'��ܻû������������ûл���@Ź������������ŹŭŢŭŵŹŹŹŹŹŹŹŹ�����������ʼڼ����ּʼ������������������������������ùìéäìòù�������� 4 ; D ! X J ? p N h Z #  9 ) B L 4 J k U @ > Q ? ? x ? * _ 8 p R e 9 * 7  D B & > x B J ] 5 s j l 3    �  �  �  �  �  �  �  �  �  b  L  6  �  �  �  �  �  �  p  �  }  2  .  �    x  �  	    p  _  �  #  n  �  5  �  +  �  �  �  �  �  D    �  �  �  a  �  ��9X��o:�o=o:�o=C�;�`B<49X<t�<#�
<49X<���<T��<T��<�h=�w<�`B<�j<u<�t�=u<��<��<���<ě�=o=��<�<�`B=�7L<�/=@�=D��<��=8Q�=C�=+=�o=H�9=H�9>   =P�`=aG�=q��=�O�=���=�v�=Ƨ�=�7L=�=���B%��B�B	��B9�Bd�B��Bb�B�B��B �HB1wB�B��B:NB3BB:B!��B��BH�A��[BцB O�B&�B�NBi�B�Bw�B\tB��B1�B �'B��BARB��B"żB)��B��B��A��B�B,��BB�B6�B��B�VB?�Bv�A�[_B��BL�B%�{B	?B	��B?�B�BXlB@8B='B��B	�B?�B�<B��B9"B@kB�B��B"6&B<eBA�A��B/�B B�B%��B�.B?*BA�BIBHB?�B2(B �B�ZBQ�B�B"��B)��B��B@eA�%�B��B,�`B�*B?�B�*BA|B<mB@�A���B<�BA`A/�wA�)�A��cA��?��AioA�ʤ@�!A?�PAxDA�S>l�A���A�9�B�cA�C�A�"KA��A��RA|�wA�w�ACt�@]S"@�)�BT]AO�A=��AinA;/�A��Ae#�A(}�A�ΌA��A�c@�ΆA"��A�"Y@��pA���C��fA�AW��C�}A�O@�]A���@���A� @���A�L*A/�A��5A�{�A�/p?�EAi KA��s@��A?kAw��A��p>�`�A���A��*B�AA�FXA�[�A�c�A�~�A}  A�W�AD�0@T�@��B�AN��AA�Ai(�A;pA�u�Ae�A('A���A��<A�d@�yzA"�^A�~8@��}A���C���A�dAZ��C�z0A�v@	�A�|�@��A��v@�-A���            3      $      	                                       .                     
      -                        !         T   
               '   *      1               !                                 !   5               '                  )         '      "                                          #      -      #                                                                  !                           #      "                                          #               N��N�1�O�3�O	�eNJc�O�uMN�.(NTfvNwJ�Nc�N;�O{��N�NN�)O���O6[N>lN��wNH��N��O���O`
8O��N�k8N��{O��O"?!N��sNg�O���NF)�O�ʡOO"~N ��N��N&?Ns��N�җN���N�+-N� 8Nd)�O�N��OZ8O���O�V�N��-NBi�O���N��f  s    �  �  �  �  �  �  �  �   �  �  �  �  �  �  u  J  �    Z  �  (  _  =  �  �  �  �  U  k    �  �  �  �  s    �  �  �    �  �    �    ]  h  y  H��`B���ͼ�t�;�`B�49X<o;o;�o;��
<o<o<#�
<t�<t�<T��<�/<�1<49X<D��<e`B<ě�<�C�<�o<�t�<���<�1<�`B<�9X<ě�<�`B<ě�<���<�`B<�`B<�<�h<�=0 �=�w=,1=��-=0 �=D��=T��=]/=]/=aG�=���=y�#=��P=��" ##0<<>=<80#""""""INQ[dgptige[PNIIIIIIXSWV]gt����������t`X��������������������ghktx�����thgggggggg���
#/<>HOOLH</#
����������������������d_`hotz~{tohdddddddd)5;=5,)(��������������������##(/5;<<<</)/6/&#������������������������������������������������������������������ $"������������

����������������������������������������������������������������EEO[ehttvth[YOEEEEEE
"099?D=<5/"ffhko{����������lhf��������������������#!#'0<IUJIA<10#####)5BLLB=50)%�������������������
"#%/<>4-)$!
��������������������������������������������������������(!)-5BBB@?5)((((((((�������������������� ')5=ACDA<5RUajntunjaUURRRRRRRR������������������������������������������������������������-./4<HIROHC<7/------A779BO[a[ZOMBBAAAAAA;5311;BHOTXUTH=;;;;;������

	��������������������������������������
���UOIH><:<<HUUUUUUUUUU����!*02)������)6BOW]VKC:6)����������������������������������likmtz~�zsmllllllll�������������������������������������������������������������������������������������������������������Óì������������������ùìàÓ�z�q�nÇÓ�6�B�O�[�b�h�i�h�b�[�O�B�6�-�)�%�)�*�6�6�L�M�Y�c�e�k�e�Y�L�G�D�L�L�L�L�L�L�L�L�L�m�y������{�y�m�`�T�G�A�:�:�=�D�O�`�m�n�{łŇŉŇ�}�{�n�b�_�a�b�j�n�n�n�n�n�n�������ļ��������������������������������Z�f�i�l�n�f�Z�P�M�L�M�S�Z�Z�Z�Z�Z�Z�Z�Z�ĿǿɿȿĿ��������ÿĿĿĿĿĿĿĿĿĿ������������������������������������������Ϲܹ������ܹϹù����������������ù��;�A�H�T�T�U�T�H�G�;�5�/�(�)�/�8�;�;�;�;�����������������������������������������\�h�uƎƔƟƤƞƚƑƁ�u�h�\�N�D�G�O�V�\�	��"�/�2�D�H�H�;�7�/�"��	���������	����������������������������������������àâìùü��úùìàÓÎÓÓÖÝàààà����(�)�(������	���������ݿ�����ݿܿѿϿϿпѿ׿ݿݿݿݿݿ��/�;�T�a�h�m�m�a�H�;�/��	������������/�Z�f�l�s���������������s�f�Z�I�D�M�R�Z��������!�"�-�0�-�!������������f�r���������������}�r�m�f�e�e�f�f�f�f���������������������������������򾱾��ʾ׾ھ�ھ׾ʾǾ��������������������(�3�A�M�Z�f�s�y������s�Z�M�A�4�(� �$�(�`�m�p�y�}�y�t�m�e�`�T�N�K�Q�T�Z�`�`�`�`�A�M�Q�W�M�I�A�4�.�1�4�?�A�A�A�A�A�A�A�A�B�[�e�g�c�[�K�5� ������������)�5�B�;�G�P�T�Z�X�T�Q�G�;�7�:�;�;�;�;�;�;�;�;����������н��������
���齷�������*�5�B�N�Z�g�t�v�t�g�[�N�B�5�)����'�*�N�N�S�N�J�A�5�3�5�8�A�J�N�N�N�N�N�N�N�N�/�<�H�T�P�H�>�<�/�&�#�#�#�(�/�/�/�/�/�/���ʼʼռμʼ������������������������������������ĽŽĽ�������������������������������������������������������������лܻ�������������߻ܻллллллл�����(�5�>�5�/�(�"����������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��l�y�����������y�q�m�l�i�l�l�l�l�l�l�l�l���ʾ׾�����"�,�*��	�����׾ξʾ���E*EEEEEEE*E*E5E*E*E*E*E*E*E*E*E*E*�#�0�<�L�V�[�Y�U�I�<�0�-�'�!���
���#�e�~�����������ʺź����������r�i�[�T�_�e���������������������������|�t�v�}�������'�4�=�;�4�.�'�������������"�'�'Ź������������ŹŭŢŭŵŹŹŹŹŹŹŹŹ�������������ʼټ�����ּʼ�����������ù������������������ùìëåìõùùùù / = D  N F ? ] N h Z )  9   : 4 J k U G > Q ? # � = M S 8 p Q e = * 7  D B  @ x B F ] 5 c j l 6    �  �  �  /  w  A  �  V  �  b  L  �  �  �  O  ~  Q  �  p  �  �  �  .  �    .  �  �  m  �  _  �  �  n  �  5  �  �  �  �  �  �  �  D  �  �  �  �  a  �  �  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  n  q  q  j  c  V  D  3    �  �  �  �  n  D     �   �   �  
          �  �  �  �  �  �  u  L      !    �  �  �  �  �  �  |  b  B     �  �  �  �  _  =      	  �  �  {  S  |  �  �    6  P  g  w    �  �  m  ?    �  �  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  C  !  �  �  X    �  �  �  �  �  �  �  d  .  �  �  1  �  L    �  9  /  �  �  �  �  |  q  f  [  L  9  %    �  �  �  �  �  �  j  Q  �  �  �  �  �  �  �  �  v  d  O  8  !  	  �       +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  _  C  (     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �   �   �          &  '  %  #  "                     �  �  �  �  �  �  �  �  �  �  �  �  �  l  G    �  �  A   �  �  �  �  �  �  �  �  {  t  i  _  U  F  2       �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  e  K  2     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  [  1  �  �  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  C  �  �  �  I  8  6  ^  m  f  _  Z  [  d  n  t  t  l  `  S  D  0  (  �  J  G  B  :  -      �  �  �  �  �  �  �  �  �    2  d  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  t  o  j  e    �  �  �  �  �  �  �  �  {  d  K  1     �   �   �   �   �   f  m  �  "  K  Z  S  C  ,    �  �  �  �  �  �  8  �  �  �  3  �  �  �  �  �  �  �  �  �    u  k  a  ^  X  M  >  +    �  (             �  �  �  �  l  E  !    �  �  �  W  �  $  _  ]  Z  W  T  P  M  I  F  @  :  2  &      �  �  �  �  �  =  1  $      �  �  �  �  �  �  �  �  ~  q  d  ?     �   �  �  �  �  �  �  v  a  H  .    �  �  �  �  Y  +  �  �  �  ]  y  w  y  �  �  �  �  �  �  �  �  �  s  G    �  �  �  {  2  �  �  �  �  �  �  �  �  �  |  f  P  9       �  �  �  g  -  m  u  |  �  �  �  �  �  �  �  �  �  �  �  i  H  &   �   �   �    ?  S  P  ?  +      �  �  �  y  g  ,  �  �    [  �  �  k  g  d  `  \  X  T  O  J  D  ?  9  4  /  ,  )  &  #         �  �  �  �  �  �  �  �  �  �  �  ~  k  S  4    �  �  �  �  �  �  �  �  �  �  �  ~  j  O  2    �  �  �  G  �  �   �  �  �  �  �  �  �  �  �  �  �  ~  {  y  v  r  n  j  g  c  _  �  �  �  �  �  �  �  �  �  �  �  q  M    �  �  r  0  �  �  �  �  �  �  �  �  �  �  z  o  d  X  L  >  0  "      �  �  s  m  h  c  ]  X  R  O  N  L  K  I  H  @  +       �   �   �  �  �  �  �  �          �  �  }  C    �  y  ;        �  �  �  �  �  �  �  �  �  �  �  r  W  8  �  z  ,  �  �  &  �  �  �  �  �  �  �  �  �  �  q  ]  E  -    �  �  �  �  V  E  �  =  �  �  :  p  �  �  �  T    �    y  �  
�  	�    d    	        �  �  �  �  �  �  �  �  �  x  [  '  �  �  �  �  �  �  �  �  w  j  ]  P  C  6  )          ,  6  ?  G  �  �  �  w  d  Q  =  (    �  �  �  �  �  �  r  Y  =            
    �  �  �  �  �  �  �  t  K  %  �  �  �  �  �  �  �  �  �  l  O  (  �      �  �  �  [  =  �  �  :  �  4    �  �  �  p  8        �  �  �  p  5  �  �  G    �    �  �  {  B  u  �  �    5  \  ;    �  �  C  �  �  �  ]  &  h  U  B  /         �  �  �  �  �  �  �  �  �  �  �  �  {  g  v  h  Q  .    �  �  �  �  �  �  f    �    �    j  �    ?  H  D  7  $    �  �  �  x  K    �  g    �  D  �  {