CDF       
      obs    5   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ə�����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       NA   max       P�#�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��      �  T   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @F>�Q�     H   (   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @v{��Q�     H  (p   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q`           l  0�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�d           �  1$   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >��      �  1�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��U   max       B0��      �  2�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��'   max       B0��      �  3�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?.S�   max       C��+      �  4t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?3k�   max       C���      �  5H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M      �  6�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          M      �  7�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NA   max       P�#�      �  8�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ᰉ�   max       ?�_o��      �  9l   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       >�u      �  :@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @F0��
=q     H  ;   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @v{��Q�     H  C\   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q`           l  K�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�d           �  L   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  L�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?�����     �  M�                           	   $                  `   �   
   L      	   6   5               	            6   	      J                  7         #         �      X      	      HNꅳN)AN�I�N�.`N$�O�I�NGO�P��NV��O�9�N��O@<uNAO^��N�ҭPB�.PN�NX3WP&�;O��Nj�P�#�P�$O{&4O� kN��WNr��N�#]OI��NccO�OO��Nt!�N���P>��N��N6N=
~N��=N� �P~v�Nv��Nб�On�NM�O�O�O�P�PsKNj|�N�,O°����㻣�
���
�o%   ;o;o;��
;ě�;ě�;ě�;�`B<o<o<#�
<e`B<e`B<e`B<�o<�o<�C�<�t�<���<���<��
<�9X<ě�<ě�<ě�<ě�<���<���<���=C�=\)=\)=t�=�P=�P=�P=�w=0 �=49X=<j=Y�=q��=u=u=y�#=�%=�%=Ƨ�=��~{zy������������~~~~����������������������������������������(*+*&*6CNOU[\a\OC6*(e`gtu|{tngeeeeeeeeeeTULLUaz��������znaVT���������������������/;AKZ^YSOH;/"�v���������vvvvvvvvvv40,6BO[anpz�wsh[OD<4������������������������������������������������������������WX\aht����������th[W���������������������������
 $%"
�������)BO[c_UH6)���^ahnt����wth^^^^^^^^!/<CUnv{whUH</#5215<AHTUaba]VMHG@<5����������������������5[glv������ugN����
/<BQ]_YM</#
�����������	
������z�����������������~z��������������������
�������
���������������������������������������������������������)5?BNVOMEB5)\bry�������������tg\������������������������������	�������������-?EE?4)��������������������������������������������#*/00/##"?BFFKOZ[hrqohe][WOB?z{�����������������z��������)NUL8������������������ #*0<HIKMI><0# #/<HRU]bY]\UI<2/%��������������������`hklqt�����������th`�������
#')&!
�������")5865)��������)4;6)��������������������������$"�����������������	�����zwz����������������z�ܹ�������������ܹչԹٹܹܹܹܽĽнݽ�ݽڽнĽ����ĽĽĽĽĽĽĽĽĽĻ������'�3�/�'��������޻����`�i�m�o�y���z�}�y�y�m�`�]�T�R�R�S�X�_�`����������������������������������������E�FF$F1F4F1F(F%F(F$E�E�E�E�E�E�E�E�E�E�ûŻɻͻ˻û������������ûûûûûûû���H�T�a�k�z�y�m�a�H�;��	�������������[�h�h�o�m�h�[�T�P�P�[�[�[�[�[�[�[�[�[�[������žϾξǾ�������f�A�(�$�+�A�Z�s��s�����������������u�s�p�m�m�o�s�s�s�s�����������������ŹŭŢŭŭŹ���������
�
������
������
�
�
�
�
�
�
�
�������ʼԼڼݼؼּʼ��������������������(�.�5�A�N�P�X�U�N�A�5�/�(�%�!�'�(�(�(�(�����������$�(�����������z�m�n�x�����ûܻ��	��������ܻû������x�b�s���ü'�4�<�@�G�@�?�4�/�'�$��'�'�'�'�'�'�'�'�ݿ���(�N�|���s�^�>�5������׿ֿٿؿ�������������������������������������������������������������������������������������C�N�f�b�H�;�,�"�����u�t�������������y�����������y�`�G�;�"�����3�A�T�`�y�/�;�H�N�T�X�`�`�\�T�H�;�/�!�����&�/�O�\�h�uƁƄƅƃƁ�u�h�\�O�C�4�0�2�:�C�OÓàæìùûù÷ìàÓÇÆÅÇÏÓÓÓÓāā�t�h�d�[�U�[�h�tāċĉāāāāāāā�Z�f�s�u�v�s�f�d�Z�M�K�D�M�Y�Z�Z�Z�Z�Z�Z�����ʼּڼ���ݼ�ּʼ����������������������ʾҾҾʾ��������������������������T�`�m�y�}�x�r�m�k�h�`�T�G�;�6�3�6�:�G�T�m�y�����ɿԿܿ�޿ӿĿ����������y�b�d�m��������������¿²©²¶¿���������������������������������������������������"�;�8�1�$�������ƳƠƑƏƞƳ�����{ǈǔǕǡǭǡǚǔǈ�{�{�o�l�o�v�{�{�{�{���������������������������������������EEEEEEED�D�D�D�D�D�EEEEEEE�#�&�/�<�H�J�U�W�`�U�H�<�/�#�����!�#�<�H�K�U�U�N�H�=�<�1�/�.�(�#�� �#�/�0�<�g�s�����������g�G�;�/�(��������5�N�g�f�s�{�v�s�q�i�f�[�Z�Y�Z�]�f�f�f�f�f�f�f�Z�f�s�x�������}�s�f�^�Z�Y�Y�U�Z�Z�Z�Z�������
����������ùòù���������뽞�����������������������������������������!�-�:�?�J�I�F�?�:�-�!��
�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�DvDiDeDlD{D��)�6�B�L�O�[�^�[�V�O�F�9�6�)�)�'�$�%�&�)�r�����������������������{�o�i�M�K�O�c�rĚĳ������������
����ĳħā�t�rāĎğĚ���#�0�4�:�0�#�������������/�<�H�U�a�b�n�n�n�k�a�U�H�?�<�/�/�#�/�/���ּ��޼ټмʼ���������r�Z�S�Y�w��� ! K k ] e < c ! V a h Z � & S ; = O = e L Z R % 2 % M I ! J ? ^ ^ L / E P D L H u x F = < B $ D F Z c Q S      +      +  �  �  �  �  �  6  �  �  �    o  �  �  /  �  �     �  �  f  �  �  ~  �  {    �  �  �  6  �  C  K  -    B  �     �  X  s    Z  s  �  �  #  ,�o�o<o;D��;ě�<���;ě�=\)<u='�<T��<�h<#�
=\)<�9X=�G�>49X<�j=�j<���<���=�t�=�hs=�w=#�
=t�=\)=o=8Q�<�`B=<j=���=\)=#�
=�/='�=�w=H�9=,1=H�9=��=@�=L��=��T=m�h=���>��=�t�>�P=�E�=�hs=��>H�9B=oB�B"�B0��B	�BC>B!zvA��UB
��B��B �B�,Bh�BYrBfB�B�9B�B�B@4B!ƛBw�B�jB�B��B!��B��BȽB"73B#v�B�^BSoBޑB��B�B��BT�B�eB7mB�BBP.B�;B%ޖB�=B*O8B�BRB+B��Bh�B�B�*B��B?�B�ZB"2MB0��B	@�B�8B!��A��'B
��B��B .�B�QB��B7�BAqB?rB��BPB��B%B!ÓB
B�BU$B�#B�B"DeB@�B�LB"DIB#] B��B\�B.B��B��B��B�\B��B�	B��BD�BLOB&-�B��B*K6B��B@HB�B�'B@B70B�(B��?.S�A(�@�jAi~�A���C��+@�qA�)Aګ�AE�lAC��A��|A�3@��0A��bA��D@���@���A��\A��{@!DeA��xAf�SA�*�B��A�RA�^!A@8@�hyAN�Ag�5Ar�	A��)A�n�B�B�hA���C�S1A�7�A�0�A��}AA��AA��A��(A!�F@o�C��HA��@��A�;�A�o�A���@�V�?3k�A(�6@��Aj�NA�~C���@��A�2�A�e�AI�AE�A��%A�W@�ܒA��A��?@�&�@�lA�w\A�r�@#�HA�Aeo�A��B+ Aˠ�A܂�A?�@�1�AN��AgIAu
)A�v�A�k�B�-B�\A�|�C�W�A�{�A���A��pACtAB*[Aѷ�A!<@tKC���A�|�@p�A�%A�U�A�T�@��                           
   $                  a   �   
   M      
   7   5               	            6   
      J                  8         $         �      X      	      I                  #      %      +                  /   1      -         M   )                           '         -                  ?                  !      '   1         $                  #      !      #                                    M   #                           #                           !                           -         N���N)AN1��Ng�aN$�O�I�NGO�O���NV��O��ON��O2@NAO'�N���O9�O��QNX3WO�M�O��Nj�P�#�O���O�Ol�dN��WNr��N�#]O�GN��O�OO���Nt!�N���O�ʾN��N6N=
~N��=N�7rO�qNv��Nб�N���NM�OG�OH��O�O}�YO��Nj|�N�ףO3׉  t  k  �  |  '  F  
    �  f  ]  f  ?  f  �  	-  �    }  �  �  �  Y  �  F    �    V  �  x  �  (  �  �  �  q  9  �  �  a  �  �  �    �  s    
,  �  �  �  	���t����
%   ��o%   ;o;o<49X;ě�<T��;ě�<o<o<T��<49X=�O�=�t�<e`B=<j<�o<�C�<�t�=\)<���<ě�<�9X<ě�<ě�<�`B<���<���=+<���=C�=�+=\)=t�=�P=�P=��=y�#=0 �=49X=}�=Y�=u>z�=u=�^5=�o=�%=���>�u�|{z��������������������������������������������������������,/6>CGOSVOC6,,,,,,,,e`gtu|{tngeeeeeeeeeeTULLUaz��������znaVT��������������������/;HLUZYTH/"	v���������vvvvvvvvvv4128BO[dinrvsh[OJA:4������������������������������������������������������������Z[_eht���������thc[Z����������������������������

���� ���)6BIJHC6) ^ahnt����wth^^^^^^^^,)((+/<HU]bfdaZUH</,5215<AHTUaba]VMHG@<5����������������������5[glv������ugN����
/<ITVTLH</#����������� �����������������������������������������������
�������
���������������������������������������������� �������������)5?BNVOMEB5)riov|�������������tr������������������������������	��������������)/021-&������������������������������������������#*/00/##"?BFFKOZ[hrqohe][WOB?�������������������������""������������������ #*0<HIKMI><0#++/:<>HRUUUHG<5/++++��������������������jlmqt|������������tj�������

�������")5865)����������
���������������������������$"��������������������������������������������ܹ����
���������ܹܹܹ۹ܹܹܹܽĽнݽ�ݽڽнĽ����ĽĽĽĽĽĽĽĽĽļ���&�&������������������`�m�u�y�z�y�t�m�`�W�V�Z�`�`�`�`�`�`�`�`����������������������������������������E�FF$F1F4F1F(F%F(F$E�E�E�E�E�E�E�E�E�E�ûŻɻͻ˻û������������ûûûûûûû��/�H�T�a�f�q�m�\�T�H�;�/�������"�/�[�h�h�o�m�h�[�T�P�P�[�[�[�[�[�[�[�[�[�[���������ȾǾ�������s�Z�M�:�7�O�Z�s����s�����������������u�s�p�m�m�o�s�s�s�s�����������������������ŹŭŤŭŮŹ�����
�
������
������
�
�
�
�
�
�
�
�������ʼμּ׼ּʼ����������������������(�)�5�A�N�O�W�T�N�A�5�1�(�&�#�(�(�(�(�(���������������������������������������������л�����������ܻлû������������'�4�<�@�G�@�?�4�/�'�$��'�'�'�'�'�'�'�'����5�N�W�`�Z�M�A�5�(������������������������������������������������������������������������������������������������C�N�f�b�H�;�,�"�����u�t�������������m�y�����}�t�`�G�;�.�"����-�@�T�\�`�m�/�;�E�H�O�T�Y�T�H�@�;�1�/�"����"�$�/�C�O�\�h�uƀƂƂƁ�|�u�h�\�Q�O�C�:�7�<�CÓàæìùûù÷ìàÓÇÆÅÇÏÓÓÓÓāā�t�h�d�[�U�[�h�tāċĉāāāāāāā�Z�f�s�u�v�s�f�d�Z�M�K�D�M�Y�Z�Z�Z�Z�Z�Z���ʼмּ޼޼ؼּμʼ��������������������������ʾʾ;ʾ��������������������������T�`�m�y�}�x�r�m�k�h�`�T�G�;�6�3�6�:�G�T�m�y�������ĿϿؿۿٿͿĿ������������m�m��������������¿²©²¶¿�����������������������������������������������������������	������������ƴƭưƻ���{ǈǔǕǡǭǡǚǔǈ�{�{�o�l�o�v�{�{�{�{���������������������������������������EEEEEEED�D�D�D�D�D�EEEEEEE�#�&�/�<�H�J�U�W�`�U�H�<�/�#�����!�#�<�B�H�S�L�H�<�;�0�/�.�)�#� �"�#�/�9�<�<�N�Z�b�g�g�a�Y�M�D�7�5�(�������(�N�f�s�{�v�s�q�i�f�[�Z�Y�Z�]�f�f�f�f�f�f�f�Z�f�s�x�������}�s�f�^�Z�Y�Y�U�Z�Z�Z�Z�����������������������������������뽞����������������������������������������!�-�:�>�F�J�H�F�>�:�4�-�%�!�����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DxD{D{D��)�6�B�L�O�[�^�[�V�O�F�9�6�)�)�'�$�%�&�)�r�~�������������������~�r�e�]�[�_�e�k�rĦĳ�����������������ĳčā�t�sāĐĚĦ���#�0�4�:�0�#�������������<�H�U�_�a�l�h�a�U�H�B�<�2�0�<�<�<�<�<�<���������üʼμ̼ʼ�����������x�~������ # K G L e < c  V i h U �  N ! & O . e L Z X ( ( % M I  L ? V ^ L ( E P D L K R x F 1 < ,  D  W c F 4    �  +  B  �  +  �  �  �  �  �  6  �  �  e  �  �  �  �  8  �  �     	  N  �  �  �  ~  G  E    %  �  �  W  �  C  K  -  �  �  �     �  X  7  �  Z  �  �  �  �  �  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  ^  k  p  b  Q  ?  ,      �  �  �  �  v  B  
  �  �  �  i  k  e  _  Y  S  M  G  <  /  "    	   �   �   �   �   �   �   �   k  �  �  �  v  g  {  �  �  �  �  �  �  �  �  �  }  k  �  �  1  [  c  l  u  {  t  n  g  `  X  Q  I  @  6  ,  "          '      �  �  �  �  �  �  �  �  �  y  c  G  +    �  �  �  F  1    �  �  �  ^  (  �  �  �  \  :  "    �  �    �  �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  �    v  l  c  �  �  �  �    �  �  �  �  g  1  �  �  R  K    �  s      �  �  �  �    `  A  "    �  �  �  �  x  q  e  B  '  ?  W    ;  S  a  e  c  Z  J  2    �  �  �  �  �  �  >  �  �    ]  [  X  S  G  ;  -      �  �  �  �  �  u  L  #   �   �   �  e  e  \  N  <  $      	    �  �  �  S    �  �  <  �  �  ?  ?  >  >  =  =  =  <  <  ;  =  @  C  F  J  M  P  S  V  Z    H  \  c  e  a  V  A     �  �  �  �  f  A    �  Z    �  �  �  �  �  �  �  w  b  P  R  C  %  �  �  �  B  �  �  S   �  �  G  �  �  �  1  q  �  �  �  	  	-  	  �  s  �  �  �  �   �  M    l  -  �  8  z  �  �  k  /  �  -  z  �  }  
7  �  �  �      �  �  �  �  �  �  �  u  ]  E  .    �  �  �  �  �  �  P  �  �    9  Y  p  |  x  a  9    �  {  �  o  �  �    �  �  �  �  �  �  �  �  t  _  I  3      �  �  �  �  d  ?    �  �  �  �  �  �  �  �  �  �  �  ~  o  `  Q  ?  !    �  �  �  �  �  �  �    h  7  @    �    ?    �  a  �  k  �  �  �  �    3  G  S  X  O  9    �  �  ?  �  o  �  e  �  %  �  Q  _  i  q  z    �  |  s  e  P  /  �  �  ~  ?    �  �  w    ,  8  @  F  D  @  7  (    �  �  �  �  Y  "  �  �    �    �  �  �  �  i  D  #    �  �  �  U  !  �  �  {  K  �  o  �  �  �  �  �  �  �  �  x  g  Z  N  D  1      �  �  �  v      
     �  �  �  �  �  �  }  d  K  9  &      �  �  �  ,  D  Q  U  U  M  =  '    �  �  �  i  3  �  �  �  C  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  j  Y  H  8  *      �  �  �  �  v  E    �  �  F  #  �  �  �  �  �  �  �  �  �    ^  )  �  �  R  �  �    g  �  z  (    
  �  �  �  �  �  �  �  �  |  j  V  <  !  �  �  �  Y  �  �  z  q  b  R  A  1  "           �          �  �  �  �    1  B  W  v  �  �  �  �  �  t  5  �  z    �    �  �  �  �  �  �  �  �  �  {  ^  B  $    �  �  �  {  V  2     �  q  m  i  f  b  ^  Z  V  Q  L  G  B  =  9  6  2  /  ,  (  %  9    �  �  �  _  )  �  �  ~  B    �  {  .  �  �  �  r  J  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  j  R  9  !  �  �  �  �  �  j  \  T  M  ?  0       �  �  �  �  �  z  ^  %  ;  3  4  /  W  _  _  `  M  -  	  �  �  K  �  Y  �  �  I  �  �  �  �  �  s  b  P  ?  -        �  �  �  �  �  _  <  �  �  �  �  {  h  T  ?  *    �  �  �  �  �  {  f  O  7     �  �  	  L  x  �  �  �  �  �  �  �    I    �  �  M    /          �  �  �  �  �  �  �  �  �  �  �  w  a  K  4    v  �  �  |  O    �  �  Y  *  �  �  �  �  �  _  )  �  �  l    �  �  �  5  �    U  r  _    v  �  �  6  �  |  g  	^  �      
    �  �  �  �  �  i  ?    �  �  �  �  a  =    �  	�  	�  	�  	�  
  
"  
*  
)  
  	�  	�  	  	*  �  P  �  �  �  �  �  �  �  �  �  a  ?  $  �  �  �  ]  $  �  �  Y    �  �  �  �  �  b  >    �  �  �  �  �  �  �  �  |  R  (  �  �  �    b  �  �  �  �  �  �  }  V  %  �  �  "  �    t  �  2  �  �  �  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	�  	s  	@  �  }  �  1  (  �  �