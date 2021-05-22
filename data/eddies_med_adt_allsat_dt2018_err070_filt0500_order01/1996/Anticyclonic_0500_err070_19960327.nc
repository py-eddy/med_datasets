CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��S���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�x�   max       P}lb       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �e`B   max       >	7L       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F
=p��     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @vy�����     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @��`           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �49X   max       >I�^       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��@   max       B08�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?c66   max       C���       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O��   max       C���       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�x�   max       P
�m       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�k   max       ?�
=p��       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �e`B   max       >	7L       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F
=p��     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�z    max       @vyp��
>     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @�-            XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?���Fs�     �  Zh      !      %         V         	                  B         (      	               /         
            	            .   �         5         2      +   !   $               )                                       C   N`;NOՋ�N���O��N�!�N��P�NA�	O2�N�rN6\Nw�jOw�O	1�N� 2P&��O-yO��O��EN���N��O/y&N�WOa
'N�a�P}lbO��SO��N
��N��yN��N/�CN�GM�x�O�OHO�k(P@cKNa1EPS�O���Nv�OtxO~]�OLiO�SO�R�O��O��>N�eCN��_N�Pp5N�¾N.n�M��FN��N�8BN�O5��O-��Ox2$N�NƶN%֣O,gN[c~�e`B�49X�o�o�o�o��o��o��o;o;D��;D��;��
;��
;��
;ě�;ě�;�`B<o<o<t�<49X<49X<49X<49X<D��<D��<e`B<e`B<�t�<��
<��
<�1<�9X<�9X<�j<���<�/<�h<�<��=C�=\)=\)=\)=t�=t�=t�=��=�w=#�
=#�
=0 �=0 �=<j=D��=L��=P�`=T��=T��=m�h=�%=�\)=�\)=���=��>	7L��������������������"5:BN[�����tg[O>:5)"����������������������������������������������������������").);>IN[gt��������t[NB;_anz����|zxnha______��������������������--..08<IIMQRNI<0----B>@BLOPVUOBBBBBBBBBBfhtt�������tlhffffff~�����������������~~!#&/4<<=<98/# sstt��������wtssssss����	/;@GPPKH<"	�����������������������}}���������������������)33/&	��dhtu��������tqhhddddY[[bhrtxwwtrhd`[YYYYqv|���������������tq������������������#/<>FHMQOG</#�������������������������/@akztZVH@��38<I[gt�����te[YOB;3����������������������������������������������������)06@BOOQOB6)"#$()#��������������������������������������� !#)/7<HJJFF?<4/#��(666/-,)���6BO_jlic[OB6)�������� )??+������������������������������2BNg���tdB5��������#%"%���!#00;:20#!!!!!!!!?8>BHN[gptttqge[NB??jjnpz�����������zrnj����
#*/1/#"
 �����**)+.9;5)�� !)5Bgsw|zvtg[N5) ������� ���������$)5@NQSSNB5+��������������}x����4029<HIUZ[UUH<444444������������������������)8@@0�����������������������)!&*6@B=6*))))))))))���������������������������	 �������;468<HHRUVUNOHD<;;;;����������������������������������������kjfmqz���������}zumkLNOT\\`cmqwz����zaTL��� ����������%)5@<5)���

�������������������
������\[^anquupnba\\\\\\\\����*�/�*������������ � ������������
���������������������������������������������������������������������)�3�:�?�A�=�3�����ܹ̹ʹϹ��Z�a�g�o�s�|�s�k�g�^�Z�N�E�D�N�O�Z�Z�Z�Z�Z�f�f�h�f�a�Z�S�T�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z�[�tĚĦĸ������ĻĳĚā�h�`�\�Z�[�T�R�[�<�@�A�>�<�<�/�/�#��#�)�/�8�<�<�<�<�<�<���� �������������������Y�f�r�������������r�f�Y�P�T�T�Y�Y�Y�Y�l�y�������������y�p�l�i�l�l�l�l�l�l�l�l�Y�Y�e�f�i�g�e�Y�N�L�F�H�L�X�Y�Y�Y�Y�Y�YƁƎƏƚƧƳƳƼƸƳƧƚƐƎƁ�Ɓ�ƁƁ�������������������y�u�m�`�T�N�T�`�m�y�����(�)�4�1�(���������������"�/�H�a�r�w�z�t�m�T�;�"�	�����������"���)�4�6�A�B�H�B�>�6�-�)��������T�a�m�z�������z�u�m�a�Z�T�L�N�S�T�T�T�T������)�3�>�B�A�6������������������޼����������������������������������������ʾ׾ؾ����׾ʾ������������žʾʾʾʿ������������ĿǿĿ����������������������s�����������������s�q�i�g�o�s�s�s�s�s�s�����)�5�-�(��������޿ۿݿ����#�/�<�?�H�M�H�H�=�<�/�*�#�!���#�#�#�#�"�;�T�^�h�c�V�H�:�/�����������������	�"���
��#�(�#��
�������������������������N�Z�g�z�r�g�_�N�A������������5�Nù������������ùöíùùùùùùùùùù����������������ݻ�����������������������������z�v�r�s�w�z���������(�1�(�������������������üȼż���������������������������������#�/�0�1�/�#�������������D�D�D�D�D�EEEEEEED�D�D�D�D�D�D�D��zÇËÓÛÓÈÇ�z�n�U�H�F�C�H�P�U�a�n�z�
��"�*�(�!��	�����������������������
���� ������ʼ������������������ּ�4�A�M�Z�c�Z�M�B�A�4�2�1�4�4�4�4�4�4�4�4�;�m�����r�T�'�� ��	����������������;�лܻ���%�&� �����лû����ûл޻߻н��!�.�4�.�.�!���������������Ľнսݽ���ݽӽнͽĽ�������������E�E�F F$F(F$FFFFE�E�E�E�E�E�E�E�E�E򾘾��������ƾ�������������s�n�l�s����������Ŀ˿Կܿ��������y�`�T�B�<�B�P�e����������$�*�,�9�A�A�5�(������������������������������ŹűŠŘŔŊŔŠŸ��ƚƧƳ��������������ƼƳƧƚƊƁƆƎƖƚ�F�:�-�!� �������� �!�-�:�?�C�F�F������������������������������������������������������������������������������#�;�S�X�b�b�U�S�I�0�����������������"�.�;�G�T�T�[�T�G�;�5�.�$�"�����"�"�S�`�l�q�q�l�`�S�Q�S�S�S�S�S�S�S�S�S�S�SÇÇÍÓÙÝÓÒÇ�~ÁÆÇÇÇÇÇÇÇÇ�������
��"���
�����������������������B�N�[�g�k�g�g�[�S�N�L�B�5�0�5�@�B�B�B�B���������������������������}�����������������ýĽý������������������������������������������������ŹŭşśŜŠŤŭ�����B�O�[�d�tāččĈā�t�p�[�O�A�6�/�)�+�B�U�a�g�n�p�w�n�k�a�\�U�K�U�U�U�U�U�U�U�U�z�{ÇÉÑÇ�z�v�w�y�z�z�z�z�z�z�z�z�z�z�\�h�i�u�x�u�h�\�O�N�O�[�\�\�\�\�\�\�\�\DbDoD{D�D�D�D�D�D�D�D�D�D�D{DqDoDgDbD`Db�����ûǻǻû��������������������������� � @ / C * g / f I . G N O L 7 1 P  g 9 A R L $ = U 6 ^ W I [ ] 6 . X 3 8 A E i ? & : 3  m [ @ 6 = " ` , D 6 � S D > = [ L � � @ B :    �     �  |  �  P  g  �  \  �  J  �  9  D  �  �  V  4  �  �  �  �  �  �  �  c  ;  �  8  �  %  X  �    E  �  D  g  {  H  �  y  8    �  H  �  #  G    �  N  �  �  A  �  �  �  �  �  �  
  s  �  G  i  c�49X<�1%   <�h<#�
%   =�{;ě�<t�<#�
<49X<e`B<49X<��
<#�
=�t�<�h<��=D��<�j<�t�<���<���=\)<�j=m�h<�j=�w<ě�<�`B<�<ě�<�<���=#�
=t�=�\)>I�^=+=u=���=�w=,1=�1=H�9=���=�O�=�t�=}�=L��=ix�=8Q�=���=D��=T��=T��=�%=m�h=�o=�+=��P=�E�=��P=���=�->0 �>�RBB�B��B��B L�B �B,UB	��B�B!�tB&F�B�^B�vBB��B̧A��B�FB ۀB�{B��B�+B2BgBL�B�B�fB{}B�wB�
B"dB��B$��B"^�BQBV�B��B��B�
BI$BpBs�B%}bB��B��B$�3B)oB�BZVB-dBzBB�tB��B B0B ��B�B�B)��B,Y�B =�A�OB*�B��B��B�jB�PB
��B	 B��B NB;BDB	�9B7�B!��B&D�B�mB�RB@GBDB�(A��@B@TB �8B�YB�nB�2B
�LBB0BA�BɥBmjB�JBґB=4B"��B��B%
�B"CZB6�BA�BDB� B�B?�B.�BC{B%�zB��B��B$@�B�.B�AB@)BsB�DB0SB�yB@�B B08�B �B��B7	B)��B,�_B C7A��rB��B;�B�9B�sB�_A��)A��A���?c66A�t6A?�A�C�A� �A0a�@�aqA�?�dBDSAlu[A4��A��A�w/A�>A��@�'AQFAsGRA�>�A��DA�A��fA�y	A��A�%q@�� A�j~A3ñ@�D�A��XC�;fAǶ�A��w@��*A;�
A��@��A�A'��C���AIYLAou�A�X�A�&�B_�@n�A���A���A�EQAb)�A]aA�N6A���A��FA �A!�LA�4A�A�A�_�A��B	�C�� @��xA�{A��A��)?O��A�~A>�A�WZA�A0�@�O�AS�?׈�B~qAm�|A4�!A���A��A�~�A�r�@�0AP��As|�A��A��tA�}�A���A�{�A��jA΀4@�+9A��nA2��@�;A���C�8Aǻ!A�rx@��
A<�xA�p3@��A7[A'[�C���AI�:As�A�L�A�vkBlJ@z��A�N)A�g�A��A_HA	RA�x�A�q�A��A ��A ��A�~'A���A�CA�YQB&0C���@���      !   	   &         W         	   	               C         )      
   	   	         /                     
            .   �         5         2      ,   "   %               )                                       C         %      !         #                           %                              7      !                           %   -      =   %               %                     '                                                                                          !                              %                                 !         -                  '                     %                                          N`;NO�N,�OY�N�y�N��Ov�pN��NC��N�N6\N5;|Ow�N���N� 2O���Nձ�N��O�\�N���N��O/y&N��OEl�N���O�|^O}�?O��N
��N��yN��JN/�CN�GM�x�N��OHOȵ�O�\�Na1EP �RO|`.Nv�OtxOo��OLiO��\O0��O
��O�$�NÁ�N�`FN�P
�mN�¾N.n�M��FN��N�8BN&�DO5��O-��Ox2$N�NƶN%֣O,gN[c~      e  ]  3  �  
Q  K  �  l  0  �  �  �  �  �  �  �  R  �  �  �  �  T  �  �  #  ?  r  h  .  r  �  �    �  �  �  �  b    �  W  
e  g  �  2  7  �  �  �  �  H    �  �  �  .  e  n  j  �    M  T  @  
-�e`B;��
���
;o%   �o=o%   ;D��;�o;D��;��
;��
;ě�;��
<���<#�
<49X<D��<o<t�<49X<D��<T��<D��<�<T��<�C�<e`B<�t�<�9X<��
<�1<�9X<�/<�j<��=\<�h=�P='�=C�=\)=�P=\)=�w=<j=@�=�w=#�
=49X=#�
=49X=0 �=<j=D��=L��=P�`=aG�=T��=m�h=�%=�\)=�\)=���=��>	7L��������������������LHINN[ggty~vtng[RNLL������������������������������������������������������������").)NPV[gt���������tg[NN`anz��zzznja````````��������������������//002<IKNNIB<0//////B>@BLOPVUOBBBBBBBBBBpjt~�������tpppppppp~�����������������~~#/3<<<<87/+#!sstt��������wtssssss�� "/7@FHG?;/"��������������������������������������)01,$
��dhtu��������tqhhddddY[[bhrtxwwtrhd`[YYYYqv|���������������tq�������������������#%/<=EHKNLHD</#�������������������������
/<HVadaH<#��>7@M[ety������th[PB>����������������������������������������������������)6;BKNB6)'#$()#���������������������������������������#"#$./<=AB<</#######��(666/-,)��)6BOVcge[OB6)	������������������������������������,04;BY\N5)#������������!#00;:20#!!!!!!!!?8>BHN[gptttqge[NB??kknsz�����������zsnk����
#*/1/#"
 ����$()(),795)��(%&)*5BN[\gf_[SNB5)(��������������������%)15?KQRRNB5.�~������������������835<HUVUUNH<88888888������������������������)8?@@5/�����������������������)!&*6@B=6*))))))))))���������������������������	 �������;468<HHRUVUNOHD<;;;;����������������������������������������kjfmqz���������}zumkLNOT\\`cmqwz����zaTL��� ����������%)5@<5)���

�������������������
������\[^anquupnba\\\\\\\\����*�/�*������������ � ����������������������������������������������������������������������������'�-�6�8�3�'�����ܹڹ޹���N�Z�g�i�p�g�Z�N�G�G�N�N�N�N�N�N�N�N�N�N�Z�f�f�h�f�a�Z�S�T�V�Z�Z�Z�Z�Z�Z�Z�Z�Z�ZāčĚġīĲİĨĦĚčā�t�q�j�h�i�k�tā�<�>�@�>�<�/�#�"�#�+�/�9�<�<�<�<�<�<�<�<�����������������������Y�f�q�r�������r�f�\�Y�X�X�Y�Y�Y�Y�Y�Y�l�y�������������y�p�l�i�l�l�l�l�l�l�l�l�L�Y�c�e�f�e�a�Y�S�L�G�J�L�L�L�L�L�L�L�LƁƎƏƚƧƳƳƼƸƳƧƚƐƎƁ�Ɓ�ƁƁ���������������y�w�m�`�V�T�P�T�`�m�y�������(�)�4�1�(���������������"�/�;�H�T�a�m�n�l�a�H�;�/�"�	�������"��)�0�6�>�B�E�B�>�6�)� ���������a�m�z�{�����z�m�a�T�O�Q�T�W�a�a�a�a�a�a�������)�0�<�?�=�6����������������޼����������������������������������������ʾ׾ؾ����׾ʾ������������žʾʾʾʿ������������ĿǿĿ��������������������������������������v�s�k�j�s�u������������������%�(�3�*�(����������޿���#�/�<�=�H�K�H�G�<�;�/�+�#�#���#�#�#�#�/�;�H�O�S�S�H�@�9�5�/�"������������/������#�'�#����
������������������������5�N�Z�g�p�j�[�N�A�(����������ù������������ùöíùùùùùùùùùù����������������ݻ�����������������������z�t�u�y�z������������������(�1�(�������������������üȼż���������������������������������#�/�0�1�/�#�������������D�D�D�D�EEEED�D�D�D�D�D�D�D�D�D�D�D��zÇËÓÛÓÈÇ�z�n�U�H�F�C�H�P�U�a�n�z���	���"�$�"���	��������������������ʼּ���������ּʼ����������������ʾ4�A�M�Z�c�Z�M�B�A�4�2�1�4�4�4�4�4�4�4�4�;�H�a�j�p�j�T�H�/�"������������������;�������� �"� �������лͻлڻ����!�.�4�.�.�!���������������Ľнսݽ���ݽӽнͽĽ�������������E�E�FF$F'F#FFFFE�E�E�E�E�E�E�E�E�E򾘾��������ƾ�������������s�n�l�s����������ɿпԿĿ��������y�`�T�F�?�E�S�i�����������"�%�)�(�(��������������Ź��������������������������ŹűŭŭŲŹƧƳ������������������ƳƧƚƋƃƈƎƘƧ�-�:�<�A�F�:�-�!���������!�(�-�-��������������������������������������������������������������������������������#�<�Q�W�U�Q�G�0��
������������������"�.�;�G�T�T�[�T�G�;�5�.�$�"�����"�"�S�`�l�q�q�l�`�S�Q�S�S�S�S�S�S�S�S�S�S�SÇÇÍÓÙÝÓÒÇ�~ÁÆÇÇÇÇÇÇÇÇ�������
��"���
�����������������������B�N�[�g�k�g�g�[�S�N�L�B�5�0�5�@�B�B�B�B���������������������������������������������ýĽý������������������������������������������������ŹŭşśŜŠŤŭ�����B�O�[�d�tāččĈā�t�p�[�O�A�6�/�)�+�B�U�a�g�n�p�w�n�k�a�\�U�K�U�U�U�U�U�U�U�U�z�{ÇÉÑÇ�z�v�w�y�z�z�z�z�z�z�z�z�z�z�\�h�i�u�x�u�h�\�O�N�O�[�\�\�\�\�\�\�\�\DbDoD{D�D�D�D�D�D�D�D�D�D�D{DqDoDgDbD`Db�����ûǻǻû��������������������������� �  . B / g " n . 6 G F O H 7 5 J  c 9 A R N # = ] 4 ] W I W ] 6 . 7 3 , / E t + & : /  n < " 6 > " ` % D 6 � S D D = [ L � � @ B :    �  +  K  �  �  P  �  q  S  �  J  N  9    �  �    �  4  �  �  �  �  �  �  m  �  n  8  �  �  X  �    �  �  �  p  {  �    y  8  �  �    �  .  6  �  �  N  i  �  A  �  �  �  T  �  �  
  s  �  G  i  c  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�    	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    d  �  �  �                 �  �  w  >  	  �  �  /  8  @  H  O  V  ]  d  d  b  V  A  )    �  �  �  z  V  3  �    .  F  U  \  [  L    �  �  >  �  �  ~  `  %  �  M    �    %  1  <  G  P  U  X  T  M  C  '    �  �  �  k  =    �  �  �  �  �  �  �  x  n  e  W  D  2       �   �   �   �   �  [  �  	I  	�  	�  
  
>  
P  
N  
B  
1  
  	�  	�  	  ^  �  �  S  |    /  ?  K  H  F  B  <  7    �  �  �  �  ^    �  �  P    �  �  �  �  �  �  �  �  �  �  �  �  �  v  W  8     �   �   �  W  _  g  k  l  k  e  ^  Q  C  4  #    �  �  �  �  ~  T  *  0  +  %        �  �  �  �  �  �  �  �  �  �  �  �  =  ~  �  �  �  �  �  �  �  �  �  �  j  P  6    �  �  �  �  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  k  b  X  N  F  @  9  .    �  �  �  �  �  �  �  �  �  �  �  �  |  f  G  )  
   �   �   �   �  �  4  �  
  Q    �  {  _  :  	  �  s  �  g  �  &  �  b  �  �  �  �  �  �  �  �  �  u  M    �  �  >  �  q  �  M  �  $  �  �  �  �  �  �  �  �  �  �  c  4    �  �  Q  �  �  2  �  :  G  R  J  =  ,      �  �  �  u  D    �  H  �    X  �  �  �  �  �  v  h  Y  F  1    �  �  �  }  U  C    �  �  �  �  �  �  �  �  �  �  �  �  h  I  +    �  �  �  �    .   �  �  �  �  �  �  �  �  �  q  [  D  ,      �  �  �  �  �  v  �  �  �  �  �  �  �  �  �  �  �  }  q  c  U  6    �  �  L  M  S  T  R  N  F  :  *      �  �  �  �  x  /  �  L  �  @  �  �  �  �  �  �  �  �  �  �  r  V  9      �  �  �  �  �  8  G  9  <  M  f  }  �  �  s  U  +  $    �  �  <  �  P  D       "             �  �  �  �  �  �  �  �  m  A     �  %  5  >  ?  3       �  �  �  ~  D    �  z  P  5  4  �  $  r  b  S  @  +    �  �  �  �  �  g  >  
  �  M  �  x  
  �  h  ]  Q  E  9  ,      �  �  �  �  �  r  K  "  �  �  �     �      %  .  .  ,  &      �  �  �  �  ]  
  �  �  Y  !  r  k  e  ^  X  Q  H  ?  6  -  "    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  6  
  �  �  }  H    �  �  {  o  d  X  K  =  0  #      �  �  �  �  �  `  7      �  �  �        �  �  �  ]  &  �  �  4  �  Q  �  _   �  �  �  w  \  ?  "    �  �  �  �  l  G    �  �  �  �  �  �  }  �  �  �  �  �  �  {  b  C    �  �  k    �  j  �  k  �  �  �  �  [  �  ,  h  ~    i  4  �  Q  �  �  �  �  	�  E  Y  �  �  �  �  �  �  �  �  �  �  x  i  Z  K  <  3  -  &      O  L  8  I  Z  U  :    �  �  �  �  �  �  �  M  2      4  M  �  �        �  �  �  N  �  �  `    �  O  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  W  K  ?  3  )        �  �  �  �  �  �  �  �  {  B   �   �  
b  
d  
T  
6  
  	�  	�  	Q  	  �  e    �  =  �  O  �  �    4  g  R  A  0    �  �  �  �  �  t  U  5      �  �  �  �  �  �  �  �  i  9  �  �  w  -    %  �  u  5  �  ]  �    6  T  �  �  �  	      1  $    �  �  �  Y  �  �  %  �  (  �  <  v  �  �  	  "  4  5  %    �  �  �  z  1  �  Z  �    V    �  �  �  �  �  �  �  g  E    �  �  �  �    ^  .  �  �  �  h  �  �  �  q  R  2    �  �  �  �  �  �  �  �  j  �  o  
  7  x  �  �  �  �  }  o  ^  K  &  �  �  �  �  V  *  �  �  �  �  �  �  }  v  m  d  [  Q  H  >  4  -  '  "            �  =  #    �  �  �  Z    �  �  "  �  Y    �  o  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  o  d  T  D  4  %      �  �  �  �  �  �  �  �  z  j  �  �  �  x  m  a  O  =  +      �  �  �  �  �  k  N  2    �  �  �  �  �  �  �  �  q  Q  /    �  �  G  �  �  W    �  .      �  �  �  �  ~  a  B  "    �  �  �  �  {  R  &   �  9  D  O  Y  `  d  a  W  I  6  "    �  �  �  �  V  #  �  �  n  i  c  \  S  H  <  .      �  �  �  �  t  J    �  �  j  j  b  \  S  C  ,    �  �  �  C    �  u  (  �  �    h   �  �  �  �  �  |  j  O  :  -  %      
  �  �  �  F  �  Y  �    �  �  �  �  �  �  �  �  �  w  f  V  E  4    �  �  �  �  M  G  A  ;  4  &    	  �  �  �  �  �  �  �  �  �    c  G  T  3    �  �  �  ^  /  �  �  �  T    �  �  ]    �  �  !  @  �  �  ]    �  �  d    �  0  �    I  
q  	i  @  �  �  �  
-  
  	�  	�  	�  	Z  	1  	  �  �  n  9    �  �  ^  #  �  �  �