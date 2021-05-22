CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Η�O�;e      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�m   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       >C�      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @Fnz�G�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vr�Q�     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >y�#      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��=   max       B1�U      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ??h\   max       C�x�      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?@�   max       C�x      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�m   max       P^�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���ᰉ�   max       ?ᴢ3��      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >V      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @Fnz�G�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @vrz�G�     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @N�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E   max         E      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*�0�   max       ?ᴢ3��     0  O�                        
         u   4      $   %      �   	      ]   %         ,   #               )   s   
      p   _             -      T                              3         �   O��oNzc�N"<9N�mO��Nm�1NTJ�N�<sN��RN��,P���O���O��P>�`P�+OZ�P��N��OR3P���O�fDO}��NX��P ��O��MO�v:O&ˏNX��Om��O�}�P��NI��OP��bP,alN�V�NS hO`թO��cO�QO���N�~@Np�N�ĢNi��O�_BN8pO)(bN�pOI�O���O	��N�;�O�&N�x��o��j��C���C��ě����
��o��o�o�o%   %   %   ;o;�o;�o;ě�;ě�;�`B<o<t�<t�<t�<#�
<49X<D��<T��<�t�<���<�/<��<��<��=o=o=C�=��=��='�=0 �=49X=<j=@�=H�9=T��=]/=m�h=q��=q��=�o=�t�=�t�=��
=ȴ9>C�"#%'*04OZbknnhbI<0("XWY[^_hontthh^[XXXX 

##%#���������������������������������������������������������������������������������������������������������������������� /<HKNamj^H</�~�����������������~���#%$*++)#
�����������
9;9,,&
�������)5N[gncVNB)�/03:<HRTSPNLHE<710//��[h������t[O6�����������������������������������������������)9;9<VWN5�����������������������)6BO[`dfhb[O=2/)#�����������������������
"/HO\^^ZUH;/"	�#!"%)5BN[gtwuj[NB5*#���������	��������?=CDO\ehu����zh\ROF?@@BBKNOVTTVSOB@@@@@@�������

�����������������������QYmz���
�����zkUQ��������������������
#/<@><:5/#
�����5VYULD5����������������������������������YSW[giqrg[YYYYYYYYYY
#/<HJRQH?</#
����
#/29<=:8/#�������������������������������
��������������������������������������������������������������������ruz���������~zrrrrrr������������
��������qnsz�������������zq��������������������QLUWV]amz|����qma[TQAEK[gtqwy�����tgQGBAKGEHN[cgmqoljg^[TNKK988=BO[[cfb[OB999999���������

�����������������������׻��������ûʻû��������~�v�l�f�f�b�l�x�������
���(����������������������	������������������'�4�;�4�4�'�$���������������*�2�@�F�@�6�*����������������� �����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�������������ݻݻ�����������������������������������������������ҽĽнݽ����������ݽнĽ��������ĽĽĽ��5�Z���������������s������޿ܿ����5���(�5�A�N�Z�`�a�Z�N�A�5�(�����������(�M�Z�d�Z�S�M�A�4�����������H�T�a�z�����������������z�a�T�H�;�5�9�H�G�m�y�}�y�w�t�o�\�J�D�;�"�������GEE*E7ECEHELECE7E*EEED�D�D�D�EEEE�����ۼݼмҼƼ������r�c�Z�g�d�V�W�x���������������������������������������������f�s�������������������s�k�f�Z�Q�S�Z�f�U�b�gŔťűŭŠ�{�b��
�������������UŭŹ����������������ŹŭŠŔŉńłņŔŭ���ʾ;ҾؾپҾʾ�������������������������������������������������������H�T�a�m�z������z�m�a�T�;�/�����9�H����������� ��������������������/�;�@�L�T�\�[�Y�T�H�/����	����"�/��"�-�.�4�3�.�,�"���	���������	�����������������ܻ������ֺ��������������ֺɺ������ɺ־4�A�M�Z�^�e�s�|�����f�Z�M�A�4�.�$�!�4čĦĸľĻĺĵĠč�t�[�?�7�.�2�A�C�O�hčàìù��ù÷ìàØÙàààààààààà���������������Ǿɾ���������������������Ƴ�������0�=�S�F��������ƱƔƍƏƙƳ�ܹ��'�9�H�S�U�L�'���ܹù������������ùܿ.�;�G�T�`�d�c�`�T�G�;�.�&�(�.�.�.�.�.�.�ݿ���������ݿտۿݿݿݿݿݿݿݿݿݿ���������������������������������;�G�T�m�y�|�����}�y�m�`�T�.�"� �%�+�-�;�����Ľ�ܽнʽ������y�h�f�a�l�p�z�������������ܻ׻û������|�~�������ûԻ��/�<�H�R�S�H�F�<�3�/�.�+�*�,�/�/�/�/�/�/�	����������
������	�	�	�	�	�	�	�	�zÇÓàèëìïìàÙÓÇÆ�z�w�v�y�z�z��������
������������������¿���������
����
��������¿µ´´»¿��(�,�)�(� ���������������(�5�A�J�N�R�S�U�S�N�E�A�:�5�4�0�(�$�!�(�A�L�N�T�P�N�A�5�,�,�5�5�A�A�A�A�A�A�A�A�{ŇŔŠŭŹŽ��żŹŭŧŠŔ�{�u�l�n�r�{�������!�.�0�"��	�����������������������T�a�m�z�����z�v�m�a�T�H�;�0�;�<�H�M�T�T�������ɺӺԺɺȺ�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDmDnDuD�EiEuE�E�E�E�E�E�E�E�EwEuEiE\EiEiEiEiEiEi G p = I 0 6 8 3 C n \ > H * R ^ [ C < @ ' 1 4 " 3 : H y : H = ; G B Q b E 1 & D N T H ' z < U b K @ K L !  k    Q  �  D  '  r  |  r  �  �  �    ]  a  .  �  U    5  �  �  �  �  p  <  ~  .  �  �  �    �  L  S    U  �  Z  �  q  �  �  �  A    �  9  Z  �  �  �  9  ;    C  ؼ��㼛��e`B�e`B<�j:�o<o;ě�<�t�;ě�=�h=P�`<�`B=��='�<���>y�#<u<��=��=8Q�=C�<���=Y�=8Q�<�`B<�1<�h=L��=�7L>z�=#�
=49X>t�>J=�w=49X=�O�=�{=y�#>J=y�#=Y�=�hs=ix�=��
=u=�t�=�o=���=��m=�E�=���>hr�>#�
B&,�B� B$a�B$�zB�SB"F�B�B! �B�B ��B.�B~B$qyB��B�B�B��B-�B �B�BB�B��B�A���B�BiB1��B�@BC�B�)B�B^!B�AB�NB��Bu�B	<BSB��B,Bc�B�?B4�B!�DB�'B3oB��B �PB��A�>�B	\�BͫB�9B�BRB&=~B��B$��B$�iB��B"O>B��B!4�B?�B B?�B?�B$>�B��B�UB��B��B<�B ��B��B?yB�B�?A��=B��BÄB1�UB�B@�B�dB��BLhB\�B�rB>!B?�B	;�B��BMB,I"BA�B��B5�B">2B=�B;�B��B J,B�:A�أB	�UB��BEXBʷB�3@�q�A0�@�qX@��A��2@�exC�x�@��FA��A+KfA�dPA�%7A5wRA�EjAd�C�z[@�T�A�J�AC�iA�a\A�S�AL��?a
�A��A�[�A��PA\��@�l@JmA=��A�K�A�~dAK�dB>??h\Ad�A.AA�%�Agl�A �h@�4�A�f�A�yA�N�@���A��0A��A��A�P�A�A�A�A�~�@&ńC��0C�1@���A4�}@��@�1�A��@�u�C�x@�&'A�}xA+A��DA�r�A2�OA��Ad�C��@��A���AC�A�	A���AL��?_p�A��MA�ǹA�OJA]X�@���@MU�A=�A�c�A̓AJ��B��?@�Ad�A~�\A�NAh�A�@�H�A��A�wyA�w�@�}2A�O�A�EtA�C�A��A�PA���A�n�@*�C�אC��                              	   u   4      %   &      �   
      ^   %         ,   #      	         )   s   
      q   `             -      T                              4         �                  !                  9         )   )      9         7            #                     7         5   -               %   !                              )                                                   !                                                      -                           %                                 #            NJ��Nzc�N"<9N�mOG�`N.�NQN�<sNG��N7}�Oa)OY@mO���O��:O���OZ�O�~+N�J#N��[O��OOP��N�NX��O�yO��-O&(�O&ˏNX��O`&O��P^�NI��OOɡ�O�iN�V�NS hO�OE~�O�QO�fEN�z�Np�N�s;Ni��O\<�N8pOLN�pOI�O��N��N�;�OE�Nn�  �  k  �  �  $  �  1  4  �  �  
W  �    �  �  �  �  �  K    �    �    V  �  �  �  J  �  
�  �     
7  	�    �  ?  S  �  a  �    )  (  U  t  �  �  �    �  0  �  
�ě���j��C���C�;o��o�o��o;�o:�o=�O�<49X;o<D��<e`B;�o>�;�`B<T��=u<��
<�C�<t�<ě�<e`B<�C�<T��<�t�<�/=�w=aG�<��<��=���=�C�=C�=��=0 �=Y�=0 �=e`B=@�=@�=L��=T��=m�h=m�h=u=q��=�o=��P=��P=��
>1'>V.-.0<GIKIE<0........XWY[^_hontthh^[XXXX 

##%#������������������������������������������������������������������������������������������������������������������������#/8<@DFGFC</#������������������������
#"(*)'#
����������"������ ��)5BN[b_VNB /03:<HRTSPNLHE<710//�����6@EE?6)����������������������������������������������� &)033,)���������������������667?BOW[^`a_[OB?:666��������������������"/;HJQSSRMH;/"
$#$')5BN[gnsqf[NB5,$����������������?=CDO\ehu����zh\ROF?@@BBKNOVTTVSOB@@@@@@�������

	�����������������������mjlq������������zm��������������������
#/<@><:5/#
��� )5:<93*�������������������������������������YSW[giqrg[YYYYYYYYYY#/<GHPNHG<0/#
#)/36863/#
��������������������������������������������������������������������������������������������������ruz���������~zrrrrrr������������
��������rotz�������������zrr��������������������QLUWV]amz|����qma[TQHDCFS[agtw������tgSHLHFJN[agkonig[VNLLLL988=BO[[cfb[OB999999�������

��������������������������ӻ�������������������������������������������
���(����������������������	������������������'�4�;�4�4�'�$��������������#�*�6�7�=�7�*�%���� ���������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٻ������������ݻݻ�����������������������������������������������ҽнݽ����������ݽнŽĽннннннн��(�5�A�Z�h�g�Z�W�N�A�5�(��������(���(�5�A�N�S�[�[�N�A�5�(������	����(�4�A�N�Z�Z�N�A�4������������a�m�z�������������z�m�a�T�H�?�?�C�J�T�a�G�T�`�i�n�n�l�f�Z�P�@�;�.�"���"�-�;�GEE*E7ECEHELECE7E*EEED�D�D�D�EEEE�������������żż������������z�z������������������������������������������������s�t���������������|�s�f�`�Z�V�Y�Z�f�s�#�0�<�I�U�c�m�i�b�U�P�<�0�#������#Ź������������������ŹŠŔŐŊőŔŠŭŹ�������ʾ̾оʾľ�������������������������������������������������������T�a�m�u�x�y�o�m�a�T�H�;�/�'� �"�*�;�H�T�������������������������������"�/�;�A�H�O�R�H�G�;�5�/�-�"�����"�"��"�-�.�4�3�.�,�"���	���������	�����������������ܻ������ֺ���� ����������ֺɺ������ɺ־A�M�T�Z�_�h�s�s�o�f�Z�M�A�9�4�/�-�4�;�A�tāĚĦİĳĴĳĪĚč�t�[�N�E�:�@�O�Z�tàìù��ù÷ìàØÙàààààààààà���������������Ǿɾ������������������������������������������ƯƨƨƯƳ���̹ܹ����'�1�3�7�4�'������ϹĹ����ӹܿ.�;�G�T�`�d�c�`�T�G�;�.�&�(�.�.�.�.�.�.�ݿ���������ݿտۿݿݿݿݿݿݿݿݿݿ���������������������������������G�`�m�t�y�{�|�y�t�m�`�T�G�;�3�1�6�;�;�G�����Ľ�ܽнʽ������y�h�f�a�l�p�z�������ûлܻ����ܻлû��������������������/�<�H�P�R�H�E�<�1�/�/�,�+�-�/�/�/�/�/�/�	����������
������	�	�	�	�	�	�	�	ÇÓàæêìïìàØÓÈÇ�z�x�v�z�{ÇÇ��������
�������������������������
����
������������¿½¿��������(�,�)�(� ���������������(�5�A�I�N�Q�R�S�Q�N�A�8�5�2�(�(�$�"�(�(�A�L�N�T�P�N�A�5�,�,�5�5�A�A�A�A�A�A�A�A�{ŇŔŠŭŹŽ��żŹŭŧŠŔ�{�u�l�n�r�{���������	��"�)��	���������������������T�a�m�z���z�t�m�a�T�H�=�@�H�O�T�T�T�T�������ɺӺԺɺȺ�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�DwDxD{D�D�D�D�E�E�E�E�E�E�E�E�EyEuE�E�E�E�E�E�E�E�E�E�   p = I  0 @ 3 K b H 7 C  J ^ 4 , ! . . & 4  2 E H y 8 B 8 ; G < H b E   D I Z H   z 6 U \ K @ D @ !  7    [  �  D  '  �  L  7  �  }  P  �  �    �  �  U  )  �    �  �    p  I  K  �  �  �  �  d  �  L  S  �  v  �  Z  ]  �  �  V  �  A  �  �  �  Z  d  �  �    �    �  c  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  P  k  d  ]  U  N  K  V  a  l  v  }  �  �  �  �  �  ~  u  m  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        #      �  �  �  �  �  j  >    �  L  �  h  �  �  �  �  �  �  �  �  �  �  �  �  �  m  S  9    �  �  �  #  $  '  /  4  7  :  :  6  $    �  �  �  {  M    �  �  |  4  2  0  -  (        �  �  �  �  �  �  �  �    .  p  �  `  �  �  �  �  �  �  �  �  �  �  �  y  V  h  B  �  �  l    i  i  i  q  ~  �  �  �  �  �    x  o  b  U  A  *       �  �  @  �  	'  	�  	�  	�  	�  	�  	�  
'  
R  
0  	�  	�  �  ?  �  �  
    f  �  �  �  �  �  u  X  8    �  �  e    �    P  q  R  �        �  �  �  �  �  �  �  �  �  s  S  9  �  J  �   �  i  �  �  �  �  �  �  �  �  �  S  )    �  �  x  Y  8  �   �  �  �  �  �  �  �  �  �  �  �  X  +  �  �  �  C  �  �  U  u  �  �  �    _  =    �  �  �  �  b  F  $  �  �  �    �  �  4  �  �  �  �      �  S  �  �  j  �  �  �  N  T  �  	#  �  e  r  �  |  t  l  c  Y  N  ?  -    �  �  �  �  y  R     �  �  �  &  <  H  J  G  ?  1       �  �  �  z  I    �  M  �  �  
  P  v  �  �  �  �  �  �      �  �      �  �  �  M  )  R  t  �  �  �  �  �  �  s  R  )  �  �  c  �  Z  �  �  N  �  �  �  �  �        �  �  �  �  �  �  K    �  J  �  d  �  �  x  i  Y  F  2      �  �  �  �  �  g  =  
  �  �  f    n  �  �  �          �  �  �  ]    �  "  e  �  �  8  L  U  U  Q  H  :  '    �  �  �  p  %  �  �  *  �  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  <    �  a    �  �  �  �  �  �  �  �  �  �  p  _  Q  Q  R  K  B  *   �   �  �  �  �  �  �  �  �       /  ;  :  3    
  �  �  �  �  �  C  H  <  (    �  �  �  �  P    �  �  m  =    �  �  U    �  �  �  �  �  �  �  �  �  �  g  %  �  �  Y    �  �  �    
  
g  
�  
�  
�  
�  
�  
�  
e  
E  
  	�  	x  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  c  Q  9    �  �  Z    �         
  �  �  �  �  �  s  R  1    �  �  �  U    �  �  �  	Z  	�  	�  	�  
   

  
  
+  
5  
  	�  	g  �  k  �  �  �  �    �  	=  	�  	�  	�  	�  	�  	�  	�  	�  	�  	f  	  �  N  �  f  �  �  6    �  �  �  �  �  �  �  �  r  _  K  9  (          /  ?  �  �  }  ^  C  (    �  �  �  �  �  l  U  F  6  (  !      �    2  ?  <  3  ,  !        �  �  m     �  |  "  �  @  #  =  <  8  2  Q  N  I  ?  (  	  �  �  a    �    �    0  �  �  �  �  �  �  �  �  k  L  *     �  �  V    �  �  f    �  9  X  `  M  +    �  �  Z    
�  
Y  	�  	N  �  �  �  �  �  �  �  �  �  �  �  �  r  Y  3  �  �  s  ,  �  �  &  �  Z   �    �  �  �  �  o  G  '    �  �  �  �  l  @    �  �  �  \    (  #      �  �  �  �  �  x  ]  =    �  �  �  O    �  (      	  �  �  �  �  �  �  �    q  c  V  I  /    �  �    1  I  U  L  ;  *      �  �  �  �  m  =  �  �  .  �    t  m  g  `  Y  S  L  F  ?  9  0  $        �  �  �  �  �    �  �  �  ~  n  ^  J  -    �  �  �  R    �  �  a    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  V  '   �   �  �  �  �  �  e  A    �  �  �  U  9  :  9    �  �  a    �  �      �  �  �  �  �  �  �  V    �  g  �  �    �  �  *  �  �  �  �  �  �  i  D    �  �  z  9  �  �  T      $  a  0  +  '  '  $          �  �  �  �  T    �  �  �  l  ,  \  �  �  �  �  �  �  �  �  y    �  �  (  .  "  �  �  
�  w  �  	�  	�  	�  	�  	�  	�  	v  	S  	)  �  �  �  R    �  T  �  8  p