CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�C+   max       Pu��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��h   max       >J      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @EZ�G�{     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vW
=p��     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @M�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >x��      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�)   max       B)��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B)D�      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?j*q   max       C���      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?o��   max       C��      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�C+   max       Pg;      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�R�=   max       ?����C�]      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >I�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @EZ�G�{     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ٙ����    max       @vW
=p��     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @M�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�?        max       @��@          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CK   max         CK      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?��J�L�     0  O�      >   	      	      `                               '   4         !      ;      U   a                  !               F               	   Z      
   4            �   4      )      	OF,PhM�N��"Nr�N�!�N|DP:�AN�O{(�N|�NDN�hO�(N��ZNIމOQk�O|��P
�kNL�"N�O�"�O��PD/IN��qP:@�O�6�N�5�N+Z�NnS}N/�JOR	�O���O$�O�*HN� �Oa�-P9��N��N��EOș�N�<�NC�6O��Nޓ:N�n�O�!vN3�O.�vNe��Pu��O�BNa��O���N��M�C+��h���
�e`B�#�
��o��o�D���o��o:�o;o;�o;ě�;ě�<t�<t�<#�
<49X<T��<�o<���<�9X<ě�<���<�/<��<��=+=+=\)=�P=��=��=��=��=#�
=#�
=#�
=,1=49X=<j=D��=H�9=H�9=T��=Y�=]/=�%=�%=��=��=�v�=�v�=�"�>Jxx����������������xx������#<QU_`H;/ ���ABBDGCLOOQV[]`[ZOBAA��������������������zxtz���������zzzzzz��������������������ADPh����������th[QEA

��������������������!#'/7<@@?></$#!!!!!!(')6ABIIB62)((((((((��������������������|������������������|not}���������tnnnnnn������


���������"'/<HUagzznkaH</%%"��������������������MKQ[g�����������ykgM��������������������T[]_dgt�����tg[TTTT��������������������1/+-5BNU[[[URXNB:511��)BNg~����tgNB:��������������������������������������������������
�������%$ 
�������������������������������������������
�����������������! ���� 
#(;<CA<7.#
 �
#+/>AHKOH</#
 �)698;;6,) b^^_chnqtxvwyyxuthbb���������������������������� 		�������{~}������������",/;<DHLSNHD;4/#"XSYWgqy���������tgX��������������������������������������������#+/-)���').58955+)������������������������������

��������������	������	 �����������������������)6BLZfmsvsk[OC)`^amoxz~zxmjca``````�����$*(�������������������������
	���
ĚĦıĳĿĿ��ĿĻĳĦĚďčČČčĖĚĚ�G�T�`�y���������m�T�;�.�!��� ��*�>�G�y���������������������������{�y�q�t�y�y�)�5�=�B�B�B�9�5�)�����%�)�)�)�)�)�)�zÇÓàìðìàÛÓÇ��z�u�z�z�z�z�z�z���!�"�-�!������	�������������ʾ۾߾ܾ׾�����}�s�Z�@�6�0�<�Z����T�a�m�n�m�i�a�T�T�K�T�T�T�T�T�T�T�T�T�T�Ľݽ����	������ݽнĽ�������������D�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�ÇÓÔÙÓÒÇ�z�y�s�zÆÇÇÇÇÇÇÇÇ��������������������������������������ٿ"�-�.�/�/�.�.�#�"��	����������	���"�L�Y�_�e�h�k�e�Z�Y�R�L�C�C�G�L�L�L�L�L�L����������������������������������������������	��������������������������лܻ�����"�������ܻлû»ûɻ��B�O�h�o�k�]�T�S�B�6��������������B������������������������������������������"�*�6�C�H�L�G�C�6�*� �������������'�3�:�?�?�<�3�'�������عֹ���T�`�m�y���{�}�y�p�m�k�`�T�G�?�C�G�J�T�T�/�H�S�L�E�H�G�9�/�"�	����������������/àìùùüùìëàÔÓÐÈÉÓÜààààù�����������	�
������ìÓ�z�k�sÆàùD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDrD{D~D������������������y�v�y�}�����������������(�2�/�)�(����� �(�(�(�(�(�(�(�(�(�(�ѿݿ����������ݿѿѿѿѿѿѿѿѿ��*�2�6�9�:�6�.�*����%�*�*�*�*�*�*�*�*�ʾ׾����	��*�)�"���	�����۾˾Ǿ��N�[�g�}�{�t�g�X�N�B�>�5�)��"�/�5�B�N¦¨­¦²µ²¦¦�L�e�r�~�����������������~�m�Y�O�@�:�E�L�_�l�x���������������������x�l�_�\�Y�_�_�(�5�A�N�Y�`�g�n�g�Z�N�A�5�0�(�%����(��������*�6�<�7�/��������������u�����˽������ĽȽĽ������������������������������#�%�/�0�<�?�<�0�#����
��
�����(�A�N�X�N�N�[�N�A�5������������<�H�M�U�Y�U�H�E�<�4�/�-�(�)�/�9�<�<�<�<��(�4�A�G�K�A�4�(��������������
�#�@�R�U�\�<�0�
��������ĺĵĳĸ�����n�{�~ŇŔŘŠšŠŞŔŇ�{�{�n�l�e�e�n�n�/�<�H�K�S�P�H�<�6�/�#��#�%�/�/�/�/�/�/�z�����������������������x�v�v�t�u�p�w�z��������������ŹŮŹ�������������������һ�!�-�=�F�S�]�S�O�F�:�-�!���������������������������x�l�`�l�x�z�����������������ʼ����� ����ʼ�������u�f�h�c����Y�f�����������r�f�Y�M�@�4�'���'�E�Y����������������źŹŷŹ����������������²¿������������¿²¦ ²E7ECEDEPEPEPECE7E1E*E7E7E7E7E7E7E7E7E7E7�I�H�<�<�<�G�I�L�U�Z�U�L�I�I�I�I�I�I�I�I , " o 3 H j + 2 Z @ ? e , A p e ' B B 5  7 I F 0 % G P h D J ! M V t I F W w , N T Z G G U b F T : K T ) \ -    +  �  �  �  �  e  $    K  �  ^  �  M  �  �    �  �  l    o  ,  �  �  (  3  �  V  �  I  �    |  P  "  �  �  �  �  �  �  w  �    �  ]  `  �  }  �  �  m    ^  �49X=,1�ě��o;�o�o=�j%   <�1<49X<o<#�
<�j<��
<T��=#�
=D��=�%<��
<�=T��<�h=���=L��=�`B>o=�w=\)=��=��=]/=�hs=P�`=y�#=T��=�O�=�G�=8Q�=D��=�+=u=ix�>O�=�\)=}�=���=}�=�-=��>x��=�h=���>	7L=�>
=qB$B��B��B��B
OxB JB2�B�eB!�]Bd?B�UB+�B7�B�[B$IB�B!�B
t{BԒB	�TB �B�6B+B"v~B��B�jB^_B�nBp�BNrB�B^-B�B��B�B\)B@�B)��A�)B	��B��B�BsVB�B=DB�B��B�FB�MBھB^�A���B6�Bc�B�AB�BDB�B@�B
@�B '�B=�B��B!@YB?�BĠB=�BB�B�	B@B�~B �#B
�eB�;B	�yB � B��B2RB"@�B�mB��BH/B��Bx�B>KB�B@�B�B?�B�B@�B?�B)D�A���B
>�B��B>vBD�B�B@B6�B��B��B� BΈBC�A���B>#B��B�AA�y�Ag�dA�`A�יA���@hJBAF݁A�F�A+�wC�MAɃ�Bc�A\��?�V&A�& A�x{@���Aנ\A�j�A��?j*qAiP�A���A˚ A���C��LAp�lA���AB3A���AXhA��A�W�@��@���A�]lA��{A"��A�~�A�D�A�qkA9mA�@�A�2cA�]�A��A���@p�@���@���@�5�A�A�C���A���A��Aj��A��A�}�A�I�@dPAF؟A�b5A+@C�F�AɃ�B>/A\��?ױ�A���A҅V@��GA֛^A�t�B :�?o��Aih+A�OoA˄AΒJC��jAp��A���A}A��{AW1�A���A�p�@�P@�LcA�~�A�{�A"��A�jdA���AÀ5A9�A��A�yA�6A�R�A�)@l�@�f-@��M@�LSA���A�|&C��A��      ?   	      	      `                               '   5         "      <      V   b   	               "               G               
   Z         4   	         �   4      *   	   	      1               /                                 )               1      +                                    /         %         )         #            5   #                  #                                                                     !                                    '                  #                        !            N�ËO�VBNqH�Nr�N.�nN|DO���N�O"D�N|�NDN�hO�(NW�{NIމN}�N��O�4NL�"N�]\O��O��O��7N��qO�ؗN���N�5�N+Z�N-�N/�JOR	�O87O$�O0bONG rO��Pg;N��N��EO
C�Nl=WN#(�O��N�cWN�n�O���N3�OH�Ne��O���O�`�Na��O��N��M�C+  w  �  �  �  &  p  	  &  �  �    c  �  �  �  �  \  .  �  �  �  �  �  L  	�  %  �  �  �    (  �  �  �  �  "  �    �  �  y  8  �    '  �  t    �  6  	�  \  �  �  �����;�o�T���#�
�o��o=,1�o;�o:�o;o;�o;ě�<t�<t�<�j<�`B<�`B<T��<���<��
<�9X=D��<���=P�`=��<��=+=C�=\)=�P=H�9=��=49X=,1=@�=H�9=#�
=,1=]/=@�=H�9=q��=L��=T��=q��=]/=��=�%>I�=�C�=�v�=\=�"�>J~�������������~~~~~~����
/<@FLI</#��BBCFIJOPT[\_[XOBBBBB��������������������|zw~��������||||||||��������������������VTUX^ht��������th_[V

��������������������!#'/7<@@?></$#!!!!!!(')6ABIIB62)((((((((��������������������|������������������|rqt��������trrrrrrrr������


���������./1<HLUVUNH<0/......��������������������ZX[agt����������tp`Z��������������������cabggt�����ztgcccccc��������������������1/+-5BNU[[[URXNB:511)5BN[`pzxog`[NB5+�����������������������������������������������

��������%$ 
�������������������������������������������
�����������������! ��
#)//00/+#

#+/>AHKOH</#

)03673+)%eaaehtvwvtpheeeeeeee����������������������������
	�������{~}������������",/;<DHLSNHD;4/#"fdafgot���������tgff���������������������������������������������)+,)���))*58954+)�������������������
	�����������

�������������������	 ���������������������������(6BJYelqtrj[O;)`^amoxz~zxmjca``````�������#('�����������������������
	���
ĦĩĳĻľĶĳĦĚĕĐĒĚĢĦĦĦĦĦĦ�m�y�����������y�m�`�G�;�,�%���.�G�`�m�y�����������������������|�y�r�w�y�y�y�y�)�5�=�B�B�B�9�5�)�����%�)�)�)�)�)�)�zÇÓàæàÕÓÇÁ�z�w�z�z�z�z�z�z�z�z���!�"�-�!������	���������������������������������s�_�U�V�f�o��T�a�m�n�m�i�a�T�T�K�T�T�T�T�T�T�T�T�T�T�ݽ�����	�
�����ݽнĽ��������Ľͽ�D�D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�ÇÓÔÙÓÒÇ�z�y�s�zÆÇÇÇÇÇÇÇÇ��������������������������������������ٿ"�-�.�/�/�.�.�#�"��	����������	���"�L�Y�]�e�e�h�e�Y�L�G�F�J�L�L�L�L�L�L�L�L���������������������������������������������������������������������������������������������ܻۻܻݻ����B�O�T�[�_�Z�S�O�I�B�6�)�������6�B������������������������������������������*�6�C�C�I�C�B�6�*�'��������������'�3�9�>�>�;�3�'������ٹعܹ���T�`�m�y���{�}�y�p�m�k�`�T�G�?�C�G�J�T�T��"�.�.�3�5�1�/�"��	�����������������àìùùüùìëàÔÓÐÈÉÓÜàààà����������� ���������ìØÒÑØàù����D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������������������y�v�y�}�����������������(�2�/�)�(����� �(�(�(�(�(�(�(�(�(�(�ݿ����������ݿԿӿݿݿݿݿݿݿݿ��*�2�6�9�:�6�.�*����%�*�*�*�*�*�*�*�*�ʾ׾����	��*�)�"���	�����۾˾Ǿ��B�N�[�g�m�s�l�g�[�T�N�B�A�5�+�,�5�8�B�B¦¨­¦²µ²¦¦�e�r�~�����������������~�r�e�_�Y�Q�O�Y�e�_�l�x���������x�l�d�_�\�_�_�_�_�_�_�_�_�(�5�9�A�N�S�Z�[�^�`�Z�N�A�7�5�)�#�#�'�(�����	�"�1�6�7�/��	�������������������׽������ĽȽĽ������������������������������#�%�/�0�<�?�<�0�#����
��
������(�5�6�A�C�D�A�9�5�(���������<�H�L�T�H�D�<�3�/�-�)�)�/�;�<�<�<�<�<�<�(�4�A�E�I�A�4�(� �!�(�(�(�(�(�(�(�(�(�(���
��#�8�H�?�0�#���������ĿĺĸĿ�����n�{�|ŇŔŗŠŠŠŜŔŇ�|�{�n�m�f�k�n�n�/�<�H�K�S�P�H�<�6�/�#��#�%�/�/�/�/�/�/�������������������������|�x�y�v�y�z�|����������������ŹŮŹ�������������������һ�!�-�:�F�S�T�S�L�F�:�-�!��������������������������x�l�`�l�x�z�����������������ʼּ������ּʼ����������������f�r���������r�f�Y�M�@�4�'���4�H�Y�f����������������źŹŷŹ����������������¦²¿��������������¿²¦¦E7ECEDEPEPEPECE7E1E*E7E7E7E7E7E7E7E7E7E7�I�H�<�<�<�G�I�L�U�Z�U�L�I�I�I�I�I�I�I�I ' 6 b 3 A j ( 2 : @ ? e , E p 1 " G B 0  7 0 F   G P q D J ' M R O @ 3 W w  V J P E G ? b B T ' J T ' \ -    �  �  �  �  _  e      t  �  ^  �  M  |  �  �  �  �  l  �  f  ,  o  �  �  �  �  V  b  I  �  6  |  �  r  d  �  �  �  +  �  8    �  �  �  `  t  }  �  �  m  �  ^    CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  CK  ,  R  e  p  v  u  m  c  V  B  )    �  �  �  \  #  �  ~  %  �  i  �  �    /  �  �  �  �  ~  m  D  �  �  �  <  �  �  n  q  �  �  �  �  �  �  �  �  �    s  f  V  G  '    �  �  u  �  �  �  �  v  j  ^  S  G  ;  1  (           �   �   �   �       #  %  &  %        �  �  �  �  �  �  E  �  �  �  �  p  n  l  j  g  e  c  a  _  ]  U  G  9  ,        �  �  �    �  r  �  G  �  �  �  �  	  �  �  �    �  �  K  f    �  &               �  �  �  �  �  �  �  �  �           M  b  v  �  �  �  �  |  p  a  E  $    �  �  u  =    �  �  �  �  �  �  �  �  �    e  C    �  �  i    �  o    �  
          �  �  �  �  �  �  �  l  P  2    �  �  �  �  `  c  Y  O  D  9  .  #         �  �  �  �  �  �  �  �  �  |  �  �  �  �  �  �  u  P  )  �  �  �  l  :  
  �  �  �  Q  M  �  �  �  �  �  �  �  �  w  _  A    �  �  }  E  	  �  �  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  ;  #  
  �  �  A  c  ~  �  �  �  �  �  �  �  �  S    �  �  (  �  }  �  �  �  �  �  �  �  #  H  \  K  +  �  �  �  p  D    �    %  u  �  �      ,  ,       �  �  V  	  �  /  �  �  
  �  �  �  �  �  �  �  �  �  ~  w  o  g  \  Q  G  >  4  +  !    �  �  �  �  �  �  �  �  �  �  �  q  A  
  �  �  F  �  �  H  �  �  �  �  t  d  U  H  <  -    �  �  x  $  �  l    V  U  �  �  �  v  e  T  B  0    
  �  �  �  �  �  �  o  S  6      =  I  X  b  z  �  �  �  �  o  M  "  �  �  D  �    a  �  L  D  8       �  �  �  a  8    �  �  f  �  p  �  �    �  �  	  	]  	�  	�  	�  	�  	�  	�  	�  	H  �  �  E  �    G  @  �  _  ,  �  b  �  I  �  �    %    �  �  m  �        	�  �  �  �  �  �  �  w  e  O  :  #    �  �  �  �  �  d  4  �  ~    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  u  m  f  ^  V  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         "  %  '  *  ,  0  3  6  9  <  ?  A  C  E  G  I  K  (      �  �  �  �  �  �  l  K  &  �  �  �  r  :  �  �   �  �  $  X  �  �  �  �  �  �  �  z  V  &  �  �    �  �    V  �  �  �  �  j  T  <  #  2  O  b  x  �  �  �  �  �  �      �  �  �  �  �  �  �  �  �  �  �  O    �  �  �  �  �  8  �  �  s  k    �  �  �  �  �  �  �  b  /  �  ~  �  g  �  �  D  �        "         �  �  �  Q    �  d    �  �  W  �  b  �  �  �  �  �  f  6     �  }  2  �  a    �  �  )    C      �  �  �  �  �  �  n  X  A  +       �   �   �   �   �   t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  E    �  [   �  �  �       @  Z  w  �  �  �  �  �  a  4  �  �  s  1  �  a  g  u  l  U  ;  !    �  �  �  e  ?    �  {    �  R  �  {  .  2  7  4  1  &      �  �  �  �  �  �  �  �  �  �  �  �  K  �  �  �  �  �  �  e  2  �  �  H  �  %  
V  	F    �  A          �  �  �  �  n  3  �  �  g    �  ~  3  �  �  \    '    �  �  �  �  �  �  p  U  9    �  �  �  U     �  �  z  �  �  �  �  �  �  �  �  �  ^  3    �  �  \  �  �  ,  �  }  t  c  R  ?  +    �  �  �  �  z  S  ,    �  �  ~  P      �          �  �  �  �  f  2  �  �  m    �  V  �    i   �  �  �  �  v  Y  :    �  �  �  �  z  Z  <    �  �  �  �  �  X  �  ?  a  �  �    -  3  5    �  �  <  �  �  �  
L  +  �  	�  	�  	�  	�  	e  	5  �  �  w    �  <  �  C  �  t  6  �  �  �  \  2  	  �  �  �  �  �  �  h  F  $    �  �  �  W  %  �  �  �  �  �  �  �  n  M  %  �  �  }  ,  �  V  �  X  �  �  )    �  �  ?  �  �  h  0  �  �  �  Q  -  
  �  �  �  �  �  F  
  �  �  v  ^  F  .    �  �  �  �  �  x  _  F  -    �  �  �