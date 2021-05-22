CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�9XbM�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�3   max       P��^      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �m�h   max       >\)      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>h�\)   max       @E��z�H     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�     max       @vL�����     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�p           �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �Y�   max       >)��      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,�(      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?3�n   max       C�3�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1xT   max       C�7�      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�3   max       PHU�      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��=�K^   max       ?ٮ�1���      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ix�   max       >\)      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E��z�H     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @vLz�G�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q            x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�@          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F   max         F      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?٫�U�=     p  S(               �            J   (   m   ;            
   C         W   
      =   	      R            C      $   &                           c   
      6   
         ,   %                     :   	   N�-1N���N�*O"x�Pf�jNW IO��NtP�O6�P��^O�CN�M�O$r6NvW�N�]�P9l�N�ON�S�OⱩM�3Oq�CPrv�NG֔O�hP9N���NT�QNm�CPGkO#0�O�q�O�%N$��O��N�^�Nk�N��NN��iN���O2�P+2�N��N�%GO��JN>�0O]��O1:POO-��N�9�N�OvUN���N-nN<�Ov�jN�qN�@z�m�h�+�������ͼ�9X��t��e`B�o��o%   ;o<o<t�<#�
<D��<e`B<�o<�o<�C�<�t�<�9X<�j<ě�<ě�<���<���<���<�/<�=o=+=+=+=+=+=C�=C�=,1=49X=49X=8Q�=<j=<j=L��=L��=P�`=T��=Y�=Y�=Y�=aG�=aG�=�\)=���=��T=���=�
==��>\)����������������������������������������ttt�������xttttttttt	
#-/6<AGC</#
(!(2M[gt������g[B5(��������������������]binz�����������zna]tttu���������yttttt���	"8@ACC@;/"���
 #/<>BD?<0#
�����5[fmkdSB5)���������������������������������������/./356AN[gjkg^[NB<5/�����������������������
 !"
���������nv����������������yna_`bntxndbaaaaaaaaaaMBJO[\hnnh[OMMMMMMMM�����
0CRSIF<0#
���<=CHJT[amz���zmaTH<��������������������emn{��������{neeeeee"/;HTaffa^TH;/%ZX\��������������iaZ926<HSUZUTH<99999999]^aanqttz{zsna]]]]]]���������������������������
#/(
��������������������������������+/.,!
���� #-<IUaa`^XQH</)# ttonst����}xtttttttt����0<IOOI<0#���dempqqt��������xwthd��������������������a`eht�����thaaaaaaaaHUaaaa[UHB=AHHHHHHHH���������������������������	! ���~���������������~������	�������
 #$$#
#/<@EHJIF?</##/02/-#���� )7CFB6&�������������������)5?AA;5)���

")56>>55)��������������������Z[ehotu����yth^[ZZZZ��������  ����������
!#*-./)#
��������������������64;;AHPRH;6666666666������������������������������������������������������������I�O�U�b�n�n�t�o�n�b�U�R�I�G�C�H�I�I�I�I�$�0�=�@�I�V�Y�a�V�I�F�=�0��$�(�$�!�$�$�zÃÇÊÇ�|�z�r�n�f�n�z�z�z�z�z�z�z�z�z���(�,�2�0�(�'������������������6�B�O�h�s�t�m�O�������������������#�/�<�H�@�<�2�/�,�$�#�!�#�#�#�#�#�#�#�#����!�+�4�;�7�.�(�����������������������������������{�r�l�r�{�����H�a�r�t�t�m�a�H�;�"�	�����������	��/�HD�D�D�D�D�D�EEEE
ED�D�D�D�D�D�D�D�D�������-�/��!������ƳƎ�`�W�Y�wƆƳ�ٻ����ûϻϻǻ������x�_�S�J�;�F�S�_�l�����"�.�;�B�C�;�5�.�"������"�"�"�"�"�"���������������������������������������ѿ����������������������������������������������������������������������������������#�<�B�A�4���齷�����}�������ݾ��(�4�?�8�4�(��������������ʾ׾����޾׾ʾ����ɾʾʾʾʾʾʾʾʺֺ����!�!�������׺ɺ����������ú�ùü����������ùìéìöùùùùùùùù����"�(�9�A�F�J�H�A�5�(���� ��������������������������g�Q�(�!�"�5�N�s��������	�����������������������������#�0�<�E�F�D�A�<�5�0�#�������������g�s�����������������g�N�7�+�(�*�5�N�Z�g�������� �����������������������������޾����������������������������������������f�s�u�s�p�h�f�Z�M�L�M�U�Z�[�f�f�f�f�f�f��)�5�B�`�e�g�c�[�@�5�)������������f�s�t�v�v�s�q�g�f�Z�M�G�A�6�7�?�A�M�Z�f�"�/�H�T�W�T�P�H�/��	���������������	�"�������������������������������|�|���������������ûƻϻû����������������������������������ϼ�����������r�f�`�]�g�r����-�:�F�S�_�l�����x�a�_�Z�S�F�C�:�-�"�"�-����������������������������������������L�Y�e�p�o�e�d�Y�L�K�B�C�L�L�L�L�L�L�L�L�g�b�[�U�N�N�B�9�7�B�N�[�g�g�g�g�g�g�g�gàìòôùýùòìàÚÚÜÙàààààà�����ʾ׾�����	������׾ʾ���������Óàù��������à�n�a�U�\�Y�@�U�n�}ÈÌÓ�l�x�����������������x�l�d�b�l�l�l�l�l�l�y�������������������y�x�m�d�m�u�y�y�y�y��"�;�T�`�\�Y�T�G�3��	�����������ݽ�����ݽҽнɽнҽݽݽݽݽݽݽݽݻ��!�-�:�=�-�)��������׺����²¿������������������������¿¸±¬®²�y�������ϿȿĿ������y�m�`�G�?�6�B�Y�m�y���
��#�/�<�F�<�9�0�/�#��
�
������������ùϹܹ�����������ܹعϹù��������������������������������������������)�6�6�B�N�[�Z�O�B�6�)�!����	���"�)ǔǡǭǹǳǭǬǡǔǈ�{�o�h�o�x�{ǈǑǔǔ�x�����������������������x�x�x�x�x�x�x�x�h�tāāāĀ�t�h�\�e�h�h�h�h�h�h�h�h�h�h�r����������������������������x�l�f�e�r����������������������������������������D{D�D�D�D�D�D�D�D{DrDoDlDoDqD{D{D{D{D{D{ < 9 7 O  G D H ? J + L * ^ D 1 W J O 3 u K d { (   7 Y k * 3 @ ' � > � N " ) f D M a D J # g V = 9 w ^ L Y u 4 | Q -    �    /  q  �  n  ?  z  �  �  �  -  �  �  y  �  �  *  ~    ;  �  ]  �  M  i  �  �  �  b  g  \  2  �  �  d  A  �  �  �  �  6  �  �  n  U  3  �  �  {  c  �  L  '  �  R  �  �  ��Y���j����49X>C��T��<t��o=�O�=�w=�G�=�+<u<�j<�t�<�j=��<���<�1=���=o=0 �=�{=+=]/=��=�P=+=\)=ȴ9=8Q�=�C�=�hs=L��=]/=Y�=��=�%=aG�=H�9=q��>t�=aG�=}�=��=y�#=�O�=���=ě�=�E�=�\)=�7L=Ƨ�=�Q�=�9X=��>&�y=�>)��B�^BB
DB�cB��B�6B�BªA���BʌBD�B"c�B�B'.BY�B�B ��B(�B=�B$�fB.�A���B�-B(��A���B
�aB#~B�@B�BB�B�{B:�B��Bn�B%oBn	B+�B4mB�SB!Z�B��B�}B�bB*5B8�B��BӽBsB��Bh)B�MB�B�Bn6B,��A�$QB��B��B�B�6BkB
?^BӰB�pB�B��B�PA��B�NBQbB"��B@B�rB?OB �B ��B()_B[oB$��B@
A��uB��B(��A��2B
�>B�<B�5B8B��B��B�XB�jB@iB%@VB�wB+��B@.BA�B!��B NB�B�PB=�B?tB�1B{JB��B9�B�BýB�B��B�;B,�(A�gwBG�B>�B?�A�ёB
��A�h�A��"A�\,A��A�n]@��A�0�C�3�B�
@���A`wBA�*uAq�VAJ��A*��A7��ARx�@F�AA��A�,�A��8@W��A�&A�	AѴ,AI��A@ A� ?A>�A��pA��o@��g@�@�$kAT�?�h�A��|A˿�AS�A�`�@�f�AoN�A^��A*��@^JA���AnrA��?3�n@�"A�1�B'P@�Q{A�);@�-�A�v�C��(A�E}B
�{A�?�A���AԀ�A�zuA�\@�*_A���C�7�B�@��A_�eA��&Aq|FAJ�=A)_A7J�AQ�(@D�A�y�A��A��@S�A�A���Aс�AJ��A? �A�n�A?A��SA�s�@��P@��;@��rAA�?���A��A�WSAS��Aʎ@@��<Ao�A^��A*�u@[S�A��Am5�A���?1xT@PA��B<�@�VmAۈ=@���A�H�C���               �            J   (   m   ;            
   C         W         >   
      R            C      $   '                           c   
      6            ,   %                     ;   
                  /            '      ?   %               3         #         7         '            %      %         #                     /                     '                                             %                  -                                    1                                                         #                     '                              N��N�NHN�*N�r*P
��NW IO���NtOD��O)F�PHU�O���N�M�O2�NvW�N�]�O:��N�ON�S�O��M�3O:�BP<�?Nq{O�hO�ABN���NT�QNm�CO�IUO#0�O��SOB��N$��Ol��N}��Nk�N�N��iN���O2�O�s�N��N�%GO��0N>�0O]��O1:POO��N���NdILOvUN���N-nN<�O=�N�qN�@z  �  9  H  �  �  �  �  �  	�  �  L  X  �  �  @  �    �  0  
|  X  �  �  �  �  	�  �  6  ]  X  G  �  z  �  u  O  !  �  9  �  �  �    H  	_  Z  �  7  !  
i  �  
  	_  �  �  �  �  F  A�ixռ�������1<�����t��T���o=C�;o=#�
<�C�<t�<49X<D��<e`B=T��<�o<�C�=}�<�9X<���=+<���<���=,1<���<�/<�=]/=+=,1='�=+=�w=�w=C�=D��=49X=49X=8Q�=��=<j=L��=]/=P�`=T��=Y�=Y�=ix�=m�h=e`B=�\)=���=��T=���=��#=��>\)����������������������������������������ttt�������xttttttttt#%/2:<@</#/,-7BN[gt����tg[NB5/��������������������_cjnz�����������znc_tttu���������yttttt	��	"&*/22//"	!#/<=AC></.#
����5BT[\UB5)�����������������������������������������0/05>BNP[gijg][NB=50�����������������������
 !"
�����������������������������a_`bntxndbaaaaaaaaaaMBJO[\hnnh[OMMMMMMMM�����
##,/)#
�����CGMT^amz~��}zrmaTRHC��������������������fmn{����{nffffffffff"/;HTaffa^TH;/%cag��������������tjc926<HSUZUTH<99999999]^aanqttz{zsna]]]]]]���������������������������

���������������������������������	$%$
�����!"#$(/<HIUVYXTLH</#!ttonst����}xtttttttt����
#0<AJHD<0#
��prrrtt}�����tpppppp��������������������gehotx���{thggggggggHUaaaa[UHB=AHHHHHHHH���������������������������	! �����������������������������	�������
 #$$#
#/<CFGHIGD</##/02/-#���� )7CFB6&�������������������)5?AA;5)���)45<=53)��������������������[[htv����wth`[[[[[[[��������  ����������
!#*-./)#
��������������������64;;AHPRH;6666666666�������������������������������������������������������������I�U�b�i�n�q�n�k�b�U�T�J�I�E�I�I�I�I�I�I�$�0�=�I�S�V�W�V�I�A�=�3�0�)�$�$�$�$�$�$�zÃÇÊÇ�|�z�r�n�f�n�z�z�z�z�z�z�z�z�z���(�+�*�(�!�������������������)�B�N�X�]�]�Q�6������������������#�/�<�H�@�<�2�/�,�$�#�!�#�#�#�#�#�#�#�#���� �*�3�9�5�-�(���� �����������������������������{�r�l�r�{�����"�/�;�H�T�[�a�b�a�[�T�H�;�/�"�����"D�D�D�D�D�EEEEED�D�D�D�D�D�D�D�D�D��������	����
��������ƜƑƍƏƗƧ���ٻ��������ûȻɻû������x�l�_�S�U�]�l�����"�.�;�B�C�;�5�.�"������"�"�"�"�"�"���������������������������������������ѿ�������������������������������������������������������������������������������нݽ�����������нĽ����������Ľо�(�4�?�8�4�(��������������ʾ׾����޾׾ʾ����ɾʾʾʾʾʾʾʾʺֺ���� ���������ֺкʺʺϺֺֺֺ�ùü����������ùìéìöùùùùùùùù���(�6�A�D�H�D�A�5�(����
����	��������������������g�Z�?�.�(�'�5�A�Z�����������������������������������������#�0�<�E�F�D�A�<�5�0�#�������������g�s���������������s�g�N�J�@�;�>�C�M�Z�g�������� �����������������������������޾����������������������������������������f�s�u�s�p�h�f�Z�M�L�M�U�Z�[�f�f�f�f�f�f�)�5�B�N�S�\�\�S�N�B�)������
���)�f�s�t�v�v�s�q�g�f�Z�M�G�A�6�7�?�A�M�Z�f��/�:�H�J�K�I�;�/��	���������������	��������������������������������������������������ûƻϻû����������������������������������������������r�k�g�f�p�v�z��F�S�_�l�x�z�x�w�l�_�S�G�F�:�F�F�F�F�F�F����������������������������������������L�Y�_�e�h�e�^�Y�R�L�G�K�L�L�L�L�L�L�L�L�g�b�[�U�N�N�B�9�7�B�N�[�g�g�g�g�g�g�g�gàìòôùýùòìàÚÚÜÙàààààà�����ʾ׾�����	������׾ʾ���������àìù��������ùìÓ�z�n�f�]�Y�]�nÇÓà�l�x�����������������x�l�d�b�l�l�l�l�l�l�y�������������������y�x�m�d�m�u�y�y�y�y��"�;�G�T�V�T�G�;�.�%��	��������	��ݽ�����ݽҽнɽнҽݽݽݽݽݽݽݽݻ��!�-�:�=�-�)��������׺����²¿������������������������¿¸±¬®²�y�������ϿȿĿ������y�m�`�G�?�6�B�Y�m�y�
��#�/�;�<�=�<�7�/�.�#���
��������
����������������޹ܹ�����躗���������������������������������������)�6�6�B�N�[�Z�O�B�6�)�!����	���"�)ǔǡǭǹǳǭǬǡǔǈ�{�o�h�o�x�{ǈǑǔǔ�x�����������������������x�x�x�x�x�x�x�x�h�tāāāĀ�t�h�\�e�h�h�h�h�h�h�h�h�h�h�r���������������������������t�r�n�p�r����������������������������������������D{D�D�D�D�D�D�D�D{DrDoDlDoDqD{D{D{D{D{D{ @ + 7 K  G A H ? K  I * Y D 1 C J O & u @ ` v (  7 Y k  3 7   � G [ N . ) f D 8 a D D # g V = * : d L Y u 4 H Q -    �  �  /  �  \  n  1  z  �  s  <  �  �  v  y  �  �  *  ~  G  ;  �  �  w  M  �  �  �  �  "  g  x  �  �  �  �  A  6  �  �  �  �  �  �    U  3  �  �  E  �  �  L  '  �  R  4  �  �  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  F  �  �  �  �  �  �  �  �  �  �  �  �  u  e  T  D  0       �  #  "  #  /  9  5  0  (  !          �  �  �  �  �  �  �  H  .    �  �  �  �  u  Q  *    �  �  �  e  ?    �  �  >  �  �  �  �  �  �  �  �  �  �  �  z  ]  <    �  �  �  ^  r  6    �  0  }  �  �  �  @  �  K  �  �  �  
�  	�  	  �  S  �  �  �  �  �  �  �  }  p  a  O  <  *    �  �  �  �  �  q  U  �  �  �  �  �  �  �  �  �  �  d  l  y  z  n  ^  M  ;  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  ^  D  *    1  �  a  �  	(  	X  	  	�  	�  	�  	�  	�  	�  	T  �  S  �  �  �  �  �  �  �  �  �  �  �  }  M    �  {  +  �  v    $  �  �  4    y  �  �    '  D  I  5    �  �  ^    �  Q  �  	  �  �  �    F  W  M  ;    �  �  �  �  �  �  f    �  ^    1  H  �  �  �  �  �  �  �  �  �  �  �    m  U  1    �  �  �  �  �  �  �  |  l  X  @     �  �  �  �  �  �  b  E  2    
  !  @  ?  >  =  3  (      �  �  �  �  �  �  �  �  ~  x  r  k  �  �  �  �  �  w  j  \  O  @  1    	  �  �  �  |  W  5    �  �  �    .  H  h  �  �  �      �  �  #  �    �  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  0  &        �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  &  t  �  	  	}  	�  
  
>  
[  
t  
{  
u  
f  
@  	�  	X  j  !  �  ?  X  �  �  �  �  �  �  �    f  M  3    �  �  �  �  �  f  F  �  �  �  �  �  �  �  �  �  �  �  q  L  #  �  �  �  G  �  �  n  �  �  �  �  �  �  ]  8  ?  �  �  /  �  j  �  2  E  Z  #  �  �  �  �  �  �  �  t  W  :      �  �  �  �  �  L  '    �  �  z  _  K  3      �  �  �  �  �  Y    �  `  �  B  �  	Z  	�  	�  	�  	�  	�  	�  	�  	^  	  �  W  �  �    �  �    �  �  �     :  3  (    
  �  �  �  �  �  `  ?      �  �  �  ^  6  8  :  ;  9  7  5  2  /  ,  (  #    &  H  j  �  �    |  ]  S  I  @  5  +       
  �  �  �  �  �  �  �  �  �  �  �    t  �  �  %  H  W  U  I  0    �  �  8  �  !  q  �  �  �  G  B  <  3  #    �  �  �  �  f  ;  *    �  �  �  E   �   y  �  �  �  �  �  �  �  �  �  �  _  H  :    �  �  K    c  >  6  S  i  w  z  v  j  U  7    �  �  n  %  �  P  �  d    @  �  �  �  �    #  -  )    �  p  (  �  �  A  �  �  B  �  �  C  =  P  k  r  t  s  i  Y  E  %  �  �  �  �  g  P    �  #  '        �  O  -    �  �  �  ~  �  �  �  �  a    �  ]  !      	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  (  Y  �  �  �  �  �  �  b  .  �  �  .  �  T  �  j  �  p   �  9  '          �  �  �  �  �  �  �  �  �  u  [  ?  "    �  �  �  �  �  �  �  �  }  s  j  `  c  n  z  �  {  k  [  K  �  �  �  �  �  z  i  Y  P  6    �  �  �  �  g  8    �    l  �  $  Q  {  �  �  f  1  �  �    x  
�  	�  �  �  �     $    �  �  �  �  �  �  �  �  o  Z  D  /    �  �  �    
  �  H  1       �  �  �  �  t  h  Z  J  +    �  �  �  �  �  a  	D  	[  	]  	Z  	S  	J  	<  	#  �  �  �  _  	  �    N  N  3  "  �  Z  V  Q  H  =  +    �  �  �  �  �  w  Y  9    �  �  �  �  �  �  �  �  �  �  �  a  9    �  �  *  �  �  �  �  �  f  @  7  2  %      �  �  �  �  [  +  �  �  �  <  �    �  o  /  !      �  �  �  �  �  �  a  9    �  �  �  .  �    �  >  
\  
a  
i  
`  
N  
3  
  	�  	�  	_  	  �    n  �  �  .  �  f  S  �  ]  �  �  �  �  �  u  ^  B  #    �  �  j    �  H  �  {      �  �  �  �  �  s  O  /    �  �  e  8  	  �  �  z  I  	_  	E  	  �  k  #  �  y     �  E  �  �  �  �  U    �  �   �  �  �  �  �  �  t  W  =  6  $  
  �  �  �  �  `  (  �  �  y  �  v  ]  D  )    �  �  �  �  �  �  �  �  o  [  G  4  !    �  �          $  )  0  7  @  Y  |  �  �  �  �  �  C  �  �  �  �  �  �  �  �  �  w  "  
�  
K  	�  	1  �  �    �  �  �  F  ,    �  �  �  �  �  d  B     �  �  �  {  L    �  �  �  A    
�  
�  
�  
N  	�  	�  	T  �  �  :  �  4  �    �  �  X  �