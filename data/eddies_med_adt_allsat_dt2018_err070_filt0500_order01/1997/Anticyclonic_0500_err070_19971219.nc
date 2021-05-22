CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�y      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
��   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       >�      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E\(��     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ə����    max       @vp�����     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O            p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >cS�      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��9   max       B+�^      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�a   max       B,0�      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�^   max       C�X�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�@�   max       C�[�      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          p      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          C      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          9      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
��   max       P� �      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�>�6z�   max       ?ח$tS��      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       >�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @E�z�G�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ə����    max       @vp�����     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @O            p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�{�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Do   max         Do      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?ח$tS��     0  O�            .   P                  	   "      *       #               K               O   L      ;      
      E         
         +   &   #            G            6      p   R   @      N=�N/�%O��Pe��P���Nχ�O��N7�Nq�tOefNO.&O�qO79O� O�)`OhqOn!�N,�SN�`�N�PB~OQ�Nn{�O���O�ZP11O�Q�O�a�P�N�OS�N��IP��N�-uN��4N�W�N��fN�dO�^5Oq�{Op��N1�N�v�N!��O��DOёNb�N
��O|mN���P��P;�O��YN��Nb�[�o���㼃o�o;o<o<t�<t�<u<�o<�o<�t�<��
<��
<��
<�j<�/<�/<�/<�h=o=o=o=+=C�=\)=\)=\)=\)=�w=49X=8Q�=8Q�=<j=<j=<j=@�=T��=�+=�O�=��-=��w=��w=�1=�Q�=���=��=��`=��=�"�=�S�=��m>+>C�>�qost�����tqqqqqqqqqq)()67=@;6+-)))))))))SOPU[gtw�����tg[SSSSHBG[h����������t[VYH�����)N^^Y5�������������
��������pht������������zz~zpjimnyz|�~zrnjjjjjjjj��������������������(),//.6BOTTOHB9760)(�����������������������������
����������������������������
>HMJNH/#������������������������ #<HUafnz�znmaU</&% ��
#/<>BA></#! �����������������������������������������016BDFEFBA;600000000G>?JSa����������naUG��������������������*6BOT[][VOJB86******��������������������		"/;?FD;4/"	���������������������������&-41)����������&+&�������� 
#/@QWZQH</����������������������������������������������������������������������
������#,05640+%# 
�������
��������������������)-5655/*)(7@BKOQ[\[ONB77777777����������� ������������#)/<9:70(#
��"/;HTWXWTQH;/"a_]amnwzzzmaaaaaaaaa��������������������94;<IUWUTI@<99999999����������������������HJJH<856<HHHHHHHHHHH! #*/33/#!!!!!!!!!!����)2/,)!���aamz��}zmaa\]aaaaaa��������������������aktz�������������tna���������		������������ ���������-,,//:<CCD</--------àìùúüùìàÔÝàààààààààà��'�0�(�'�����������������N�[�g�t�v�w�u�t�n�g�[�N�L�G�D�H�N�N�N�N�Z��������¾�����Z�M�>�F�;�������(�ZĦ���#�I�o�{�j�[�V�#�
��������ĳĘĊčĦ�����(�+�(�"������������ ���ѿݿ�������ѿÿ����y�~�����������Ŀ��z�������������������z�w�z�z�z�z�z�z�z�z�Ϲܹ����������ܹϹĹȹϹϹϹϹϹϻûɻлܻ��	�����ܻӻû������������û����	���#����� ���������������'�:�@�Y�e�`�T�M�@�4���������������)�3�/�2�/�)������������������<�a�p�{ÅÓâàÓÇ�~�l�_�H�#����#�<àìù����������������ùàÓÇÂÁÅÇà�����	����������������������������m�y�����|�w�e�`�T�G�;�9�6�.�.�4�G�T�`�m�������������������������������������������������������������Ҽ��ʼ̼̼ʼ������������������������������5�N�Z�s���������������Z�N�/�����%�5���������������������������������������������������|�z�s�m�k�m�q�z�}�������������������������������������g�N�D�L�Z�g�s���H�T�a�k�p�m�e�T�;�/��	�������	��/�;�H�a�m�z���������������z�m�T�;�4�(�*�1�T�a�`�m�y�������Ŀ��������m�`�S�G�B�C�G�T�`�������3�5�)�������ݿӿѿſѿݿ��"�&�;�H�R�Y�[�[�T�;�/�������������	�"��!�)�!�!����������������`�m�y���|�y�m�`�T�G�;�.�"����.�;�G�`�s��������������s�q�n�m�n�s�s�s�s�s�s�������ʾ����	����ʾ������x�z������:�:�9�.�)�!����������������!�.�:�ʾʾ������ɾʾ׾�����������׾оʾʹ��������������ܹҹϹʹϹܹ����h�uƁƎƒƎƅƁ�x�u�h�h�\�O�N�O�\�`�h�h���������������������������������������������������ƽǽĽ��������y�l�`�\�`�m�y��������)�5�5�7�5�)������������������$�/�6�;�?�6�0�#���
� ����������
��$ÇÓàéáàÓÈÇÆÂÇÇÇÇÇÇÇÇÇ���������������������������s�j�o�s������ֺ���������ںֺӺֺֺֺֺֺֺֺ�DoD{D�D�D�D�D�D�D�D�D�D�D{DoDbD\DZD[DbDo�I�V�b�k�n�j�b�V�Q�I�=�8�0�-�0�6�=�D�I�I�o�b�V�R�P�V�b�o�w�o�o�o�o�o�o�o�o�o�o�o���
��"���
��������������������������������+�7�A�B�I�B�6����������������������������ŹŭŨŤŭŭŹ�������������Ҽr��������ֽ�������ּʼ����q�d�f�r���!�-�@�H�P�R�N�:�-�!������޺ֺѺ�²¿�������
�����
��������¹¤²E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����*�6�7�6�6�*������������ 1 g 0 > H F R \ s V ` R " x < T 3 9 ; f * S s E ] % & 3  X | V J : Y - V m 1 B 4 Q 0 S & 9 C e V $ ] W R k M    N  :  ;  �  �  �  �  e  �  f  �  �  �  �  0  '  �  N  �  B  J  �  �  �  �  �  -  N  H  A  .  �  �    �  �  �  k      �  Z  �  Z  �  &  /  '  =  �  
  �  �  �  v���ͼ�o���
=��=��T<u<�j<49X<�1=C�<ě�=T��=<j=}�=T��=ix�=T��<�=#�
=C�=��=aG�=��=Y�=q��=�x�=�G�=u=�v�=49X=]/=D��=�x�=Y�=P�`=e`B=]/=ix�=�;d=�"�=�S�=��=�v�=\>$�/==�;d=�;d>�w=��>cS�>Q�>H�9>�>+B
!uB�0B	a�BP�BY�B#sB �B�+B�gB�5B)�B"�Bd#B�!B!]wB�B�AB�8B~�B�Bk@B BA�B�A��9Bv�BY�B;lB�rB��B!�!B�eB�|B%#�B$�B ��B�5Bz�B+�^B3�A���A�\%BNB&�+B�gBg�B�7B��B��A���Bo�B��BB6�B��B
6(B�B	�B@3B�B#BVB��B;B�PB_�B?�B":BE�B��B"G�B�,BD/B|�B��B�B[�B��BK_B��A�aB��BE�BAB0^B��B!@B��B��B%=[B$leB @~B��B�;B,0�BABA���A��B?�B&�B��BKPB�RBȩB�KA���B��B9xB<�BK�B�yA�X�?���A���AC�@A��A0�AwIA���?�^@�5@��+@�rAԼ�A��A̼�A�mAg��A�-A�sM@��KA�K�A�{A��JA�A�bA�i�AmA��A��/@e�FAd�4AC��AP�XA;AT\{?LGB�R@ NA�(A�p�A鸅AʆMA��@D�C��>BteB>�A�_A�ZA��@��O@^�A��C�X�A��A�s�?�{�A���AD��A�A/oAu 1A�#?@���@���@�;�A�gAƀ�A� �A�}�AigA�G�AЂ@���A���A��XA��XA��A��}A��DAl��A�iqA���@c��AgS�ACW
AQ�AB0AT=>�@�B}�@$)�A��A�M�A�zA��A�x�@D�C��BA`B?jA�ݛAӱ�A��A ��@l"A� C�[�A�~�            /   P                  	   #      +   !   #               K               P   L      ;            F                  ,   '   #            H      	      6      p   S   A                  9   C      #               #      +                     +               '   #      %            '                  %                        
            -   )   "                     9      #               !                           '                        #                                                      
            !   )   "      N=�N/�%O��OMb�P� �N8�]O��N7�Nq�tN�)�NO.&O�KN�<O�&�O}0MN���On!�N,�SN�`�N�P�IOQ�Nn{�O�ӎO��O��2O�rpO�[�O��N�OS�N��IO��N�-uN��4N�W�N��fN�dOe�eOQWWOp��N1�N��N!��O��DOёNb�N
��OH	N���O��>P;�O��YN��Nb�[  o     �    �    �  �  �  �  �  �    n  e  r  j  �  �  �  �  Y  �  9  �  	�  
7  @  '  h  |  0  	%  G  =  8  $  	  ,  w  �  �  �  �  f  `  �  �  v    &  %    �  ��o���㼃o<���<�C�<49X<t�<t�<u<�t�<�o<��
<�/<�`B<�1=�w<�/<�/<�/<�h=8Q�=o=o=t�=�P=�%=u=�P=0 �=�w=49X=8Q�=�7L=<j=<j=<j=@�=T��=���=�t�=��-=��w=���=�1=�Q�=���=��=��`=�G�=�"�>O�=��m>+>C�>�qost�����tqqqqqqqqqq)()67=@;6+-)))))))))SOPU[gtw�����tg[SSSSZZ]dht��������xthc_Z�����)BNVVP5����������� ������������pht������������zz~zpjimnyz|�~zrnjjjjjjjj��������������������)-01016?BMORROLFB6))����������������������������������������������� ������������
#+'#
���������������������������./0<HKTOH<1/........��
#/<>BA></#! �����������������������������������������016BDFEFBA;600000000GDEU^z���������znaUG��������������������*6BOT[][VOJB86******��������������������
"/;=DB?81/%"��������������������������#'(%�����������$)$�������#/<HSVRL</
��������������������������������������������������������������������
�����#,05640+%# 
�������
��������������������)-5655/*)(7@BKOQ[\[ONB77777777������������������������#/584//'#
��"/;HTWXWTQH;/"a_]amnwzzzmaaaaaaaaa��������������������94;<IUWUTI@<99999999����������������������HJJH<856<HHHHHHHHHHH! #*/33/#!!!!!!!!!!�����)*#���aamz��}zmaa\]aaaaaa��������������������aktz�������������tna���������		������������ ���������-,,//:<CCD</--------àìùúüùìàÔÝàààààààààà��'�0�(�'�����������������N�[�g�t�v�w�u�t�n�g�[�N�L�G�D�H�N�N�N�N�s�������������������p�f�U�M�K�M�Z�f�sĳ����<�\�b�N�H�>�#�
��������ĿĦęĠĳ����������������������������������ѿݿ�������ѿÿ����y�~�����������Ŀ��z�������������������z�w�z�z�z�z�z�z�z�z�Ϲܹ����������ܹϹĹȹϹϹϹϹϹϻŻлܻ�����������ܻ׻лû������Ż����	���#����� ���������������8�@�I�M�X�[�O�@�4��������������)�-�,�.�)����
��������������a�n�wÀÄÁ�z�n�i�H�<�#����#�<�H�U�aàìù����������������ùàÓÇÂÁÉÓà��� ���������������������������������m�y�����|�w�e�`�T�G�;�9�6�.�.�4�G�T�`�m�������������������������������������������������������������Ҽ��ʼ̼̼ʼ������������������������������A�N�Z�x�����������s�Z�N�5�&����"�-�A���������������������������������������������������|�z�s�m�k�m�q�z�}�����������������������������������g�Z�Q�S�]�g�s�����/�;�H�T�a�i�n�j�b�T�;�/�"��	���	���/�z���������������z�m�a�T�H�B�F�M�T�a�m�z�m�y�����������������y�m�`�W�O�P�V�`�c�m�������(�1�'��������ݿտҿݿ����/�;�H�T�W�V�P�A�/��	������������	���!�)�!�!����������������`�m�y���|�y�m�`�T�G�;�.�"����.�;�G�`�s��������������s�q�n�m�n�s�s�s�s�s�s�����ʾ����������׾ʾ����������������:�:�9�.�)�!����������������!�.�:�ʾʾ������ɾʾ׾�����������׾оʾʹ��������������ܹҹϹʹϹܹ����h�uƁƎƒƎƅƁ�x�u�h�h�\�O�N�O�\�`�h�h�������������������������������������������������������������y�s�m�o�t�y�}������������)�,�2�1�)�������������������$�/�6�;�?�6�0�#���
� ����������
��$ÇÓàéáàÓÈÇÆÂÇÇÇÇÇÇÇÇÇ�s���������������������s�k�p�s�s�s�s�s�s�ֺ���������ںֺӺֺֺֺֺֺֺֺ�DoD{D�D�D�D�D�D�D�D�D�D�D{DoDbD\DZD[DbDo�I�V�b�k�n�j�b�V�Q�I�=�8�0�-�0�6�=�D�I�I�o�b�V�R�P�V�b�o�w�o�o�o�o�o�o�o�o�o�o�o���
��"���
���������������������������������'�2�6�;�7������������������������������ŹŭŨŤŭŭŹ�������������Ҽʼּ���������ּʼ����������������ʺ��!�-�@�H�P�R�N�:�-�!������޺ֺѺ�²¿�������
�����
��������¹¤²E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����*�6�7�6�6�*������������ 1 g 0 ' I @ R \ s O ` P  N > & 3 9 ; f ! S s @ M   ,  X | V 2 : Y - V m 9 9 4 Q - S & 9 C e I $ O W R k M    N  :  ;  �  �  F  �  e  �     �  �    n  '  �  �  N  �  B  �  �  �  C  ;        �  A  .  �  ,    �  �  �  k  �  �  �  Z  �  Z  �  &  /  '  �  �  �  �  �  �  v  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  Do  o  l  h  b  Z  Q  D  3  #    �  �  �  �  �  �  �  n  [  H     �  �  �  �  �  �  �  �  �  s  c  R  ?  %    �  �  �  �  �  �  �  �  �  }  g  Q  :  !    �  �  �  x  P  &  �  �  ]  z  �  !  T    �  �  �  �    
      �  �  �  5  �     �  �  �  �  �  �  �  �  �  Z  �  �    �    �  d  3  �  n  ^       	                                 1  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  T    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  p  e  [  Q  �  �  �  �  ~  u  m  m  s  x    �  �  �  �  �  �  x  b  L  n  �  �  �  �  �  o  Y  @  !  �  �  �  w  Z  D  !  �  �  y  �  �  �  �  �  �  �  �  �  �  �  �           �  �  �  �  �  �  �  �  }  {  �  }  m  _  N  0      �  �  �  �  �  �  �                �  �  �  �  �  Y    �  i    �  t  �  �  �  )  i  m  e  U  @  .      �  �  �  �  a  �  t  �  �  [  e  a  V  D  E  E    �    5  �  �  �  d  *  (  a  8    �    ]  �  �    9  U  i  q  b  I  '  �  �  p    �  =    j  W  E  5  &    	  �  �  �  �  �  ]  3    �  �  B  �  "  �  �  �  �  |  v  p  j  d  ]  W  P  J  G  J  M  P  T  W  Z  �  �  �  �  �  �  �  �  �  v  R  +     �  �  c  +  �  �  {  �  �  �  �  �  �      !        �  �  �  �  �  A     �  )  X  v  �  ~  k  J  #  �  �  n    �  [  �  \  �  �  �  �  Y  N  @  0      �  �  �  �  T  !  �  �  u  0  �  �  �  Q  �  w  l  `  X  P  H  B  ;  5  6  ?  H  J  A  7  /  /  .  -  /  4  7  8  5  ,      �  �  �  Y  7    �  �  w  
  �    �  �  �  �  �  �  �  W  (  �  �  j    �  o    �    p  �  f  �  �  	  	F  	u  	�  	�  	�  	W  	  �  �  ]    �  �    �  |  �  	X  	�  	�  
  
.  
7  
4  
&  
  	�  	�  	-  �    Z  �  �  '  �    7  ;  2  #    �  �  �  �  X  )  �  �  �  7  �  W  �   �  �    $  &    �  �  �  �  n  3  �  �  D  �  �  7  �  �  <  h  `  X  P  G  <  1  &      �  �  �  �  �  �  �  r  `  M  |  i  W  L  B  ;  3  (        
  �  �  �  �  �  n  /   �  0  .  +  )  &  $  "          �  �  �  �  �  �  �  �  �    k  �  �  	
  	  	%  	   	  �  �  �    �    �  �  B  q  |  G  :  -         �  �  �  �  �  �  �  �  �  g  H  #   �   �  =  5  -  %          �  �  �  �  �  �  �  �  �  p  [  E  8      �  �  �  �  b  ?    �  �  �  n  /  �  �  �  �  �  $      �  �  �  �  �  �  �  �  �  �  }  r  e  Y  L  ?  3  	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  f  �       �  �       ,  *    
  �  �  �  X    �    }  �  '  u  u  q  e  U  =    �  �  �  m  6  �  �  -  �    l  
  �  �  �  U  '  �  �  5  �    !  �  �  T     �    q  �    �  �  �  �  �  ~  {  y  v  t  o  h  b  [  T  L  D  <  4  ,  �  �  �  �  b  8  
  �  �  v  ?    �  �  ]        �  �  �  g  N  ?  -      �  �  �  �  �  d  >    �  �  �  �  �  f     �  ~  $  �  �  >  �  �    
�  
  	f  �  �      �  k  `  D  &    �  �  r  6    �  �  [    �  �  S  &  �  {  �  �  w  N  %  �  �  �  �  �  �  x  _  E  $    �  �  r  ?    �  �  �  �  �  �  s  c  T  D  5  $    �  �       $  "     t  [  \  g  9  �  �  g    
�  
`  
  	�  	  �  �  ;  w  �      >  �  �  �  �  ^  :    �  �  �  c  4    �  �  l  8  �    �  �      #    �  �  �  �  �  a  �  <  
G  	  k  �  n  %      �  �  �  �  q  4  
�  
�  
  	�  	  k  �  q  o  v  �    �  �  �  N    �  n    �    
�  	�  	"  X  s  ]  �  �  �  �  �  =  �  �  �  `  (  �  �  �  Y  +  �  �  �  �  d  3  �  �  N  %    �  �  �  x  O    �  �  W  �  �    �  /  �  6