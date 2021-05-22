CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ͳ-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�B'   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �#�
   max       =�9X      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F��
=p�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vr=p��
     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       >�V      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4��      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�Sg   max       B4��      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?@GK   max       C�u�      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�d   max       C�w�      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          '      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�B'   max       O��      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?��TɅ�p      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �#�
   max       >\)      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @F��
=p�     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @vr=p��
     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @ꃀ          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Cg   max         Cg      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�$tS��N   max       ?࿱[W>�     `  U�      +      A                               	                           s   G                                          J                     Y   &                     �                     !   	   	NT�O�x�N�qCP�zN�w�M�B'NgN�O�E;O��O�!cM�;O _�OB?DN�*�NP)�N��O�WcN3�O��N!ZNQFHO
twP��O�z#O:sN
�5OO�NF�N���O��OKO>9�O]��N��O1ϖNv{O �'P3p�OC�JO1��ON��N��TN��)N�VVP�(O�<�O'�YN�*SO_��N�әO5"nN��PxW�M��O6$�N%<�O��OM��ORW�ObҧNg��N����#�
�e`B��o�o;o;D��;��
;��
;ě�;ě�;ě�;ě�;�`B;�`B<t�<t�<49X<D��<T��<T��<�o<�o<���<���<���<���<�1<�1<�9X<ě�<���<���<�`B<�`B<�h<�<��<��=+=+=\)=\)=\)=t�=��=��=��=#�
=#�
='�=8Q�=<j=@�=H�9=P�`=Y�=aG�=�+=�\)=���=���=�9X��������������������@=@BN[gt������~tg[N@��������������������4+;A?@EOht�����tkaB4��������������������~}���������~~~~~~~~)3585)!/3<Un{������{nUI0*!)5:?ABGB:5)
#/<U]abZUH<(#����������������������������������������	"/;HOHC;8/"	����������������������������������������/6BOS[\[OB;6////////afv�������������vhamdnz����znmmmmmmmmmm��������������������$)26BEEB?61)$$$$$$$$		$������������������~z������'"������������������������������������#'%���hkttv����thhhhhhhhhh����������������������������������������15BN[gtttrgb[QNB;851����#/7:9<@<:/#
���������
������������#(���������	������������������������)5BELKB54)))595/))��������������������(Oht�|q[OJ6)� 	#0<?ED>80$"
��dbach�������xtqkmlid #+*/158:7/#'*+-/2<ABB><4/''''''93:<AHUY_VUHG<999999gmkmz���������zmgggg������  ������������������������������� !���% %)45BNVNLFBA53)%%U]]`egt���������qg[U{}�����������������{������ �������))11/)�������)6OF.������eegty|~tsgeeeeeeeeeez������������}zwzwwz��������������������MLTV`]emz����zpmaWTM)16<BO[`bbOB6)��������������������������������������������������������
!#-/0/###
����� ���
��������������U�nÇËÑÕ×ÔÙÓÇ�z�s�a�U�J�C�A�E�U�#�/�<�H�Q�O�H�?�<�0�/�,�#����#�#�#�#���M�s�����ʾ׾��ྱ���s�b�(��������)�6�@�6�.�)������������������� �������������������������������޾f�o�o�p�f�b�Z�V�R�S�Z�a�f�f�f�f�f�f�f�f�����������������������w�v�q�q�d�a�^�f�����ʾ���������׾ʾƾ����������������m�y���������k�G�;�.�"�������.�G�m�����	�����������������������������x�������������������x�s�l�l�x�x�x�x�x�x�T�a�g�m�r�w�y�y�r�m�_�T�K�H�=�>�B�H�P�T�����ʼμּռʼ��������������������������'�0�3�;�9�3�'�"�����'�'�'�'�'�'�'�'�����������{�z�y�u�t�z��������������������������������������s�f�Z�M�Z�f�s������/�<�=�>�=�<�/�+�%�/�/�/�/�/�/�/�/�/�/�/�������������������������������ž����ÓÖàäåàÓËÇÂÇÐÓÓÓÓÓÓÓÓ�6�C�O�\�h�k�q�h�\�R�O�C�6�3�6�6�6�6�6�6�/�4�;�>�<�;�5�/�"���	�	�	����"�*�/Ó�������������������ìà�c�X�W�Y�aÄÓ��5�A�U�_�_�Z�N�A�5�(�#���������5�9�A�N�Z�g�s�~�������s�g�Z�N�C�9�3�2�5ù������������ùøñùùùùùùùùùù����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ������������������y�p�m�g�`�^�`�m�y�����)�8�;�9�?�>�5�)�������������������x�{�w�x�y�}�}�x�l�_�S�F�<�>�F�O�W�_�p�x���������ĿʿпĿ�������������������������"�.�:�V�j�`�T�F�;�.�"�	����������	���"�"�"�"����	��������������	�	ŔŠŭŹ����������������ŹŭšŠŔœŊŔ�����������������������������A�M�T�Z�\�]�Z�M�A�:�4�(�"�����(�4�A�����ûܻ��9�M�Q�4�'���ǻ������y���������� ���������ڼּּڼ���𼤼����ʼм˼������������r�e�f�r����������(�5�A�K�N�Z�c�Z�N�C�5�(���������s���������������������|�s�g�s�s�s�s�s�s���(�3�(�'� ���� ����������h�tāĄČčĎĎčā�t�p�h�c�]�a�h�h�h�h������(�-�,�#�����¿¦£ ¦²¿��������������������������������������������������"�/�5�9�0�/�"��	������������������	���"�#�%�'�"���	���������������/�;�H�T�a�c�h�d�a�T�H�;�/�"���"�'�&�/�ʾ׾۾�����������׾ҾҾоʾľžʽ��������������~�y�l�k�d�`�]�[�`�b�l�y�����������������������������������r���������������ּ������u�n�m�w�l�b�r�O�[�\�^�[�O�B�B�B�M�O�O�O�O�O�O�O�O�O�O�C�=�8�2�)�%���������������)�6�C�����������������������������������������{ŇŐŔŠŭ��������ŹŭŠŘŔőń�{�v�{�������������ɺ׺ֺȺ��������������������������������������������������������������������������������}�s�Z�A�6�A�N�g�s�����(�4�-�(� ����	���������ǡǭǯǱǭǭǡǙǔǐǈǈǈǎǔǝǡǡǡǡ _ 3 3 [ W w X ] D ] G  4 ; @ | / s X K � @ 6 + D C V A _ ! W ' _ [ = b K n 3 f S o M I F  = T 1 o Z * e K 4 R ~ | E { ; I  8  t  �  �  �  E  �  '    9  /    �  �  n  �  �    @  U  �  H  �  �  �  ;  �  ^  ^  �  �  �    C  �  �  7  �  �  �  �  �  �    �    �    �  �  �  �  �    �  4  �    �  _  l  ���P<�`B<#�
=�%<T��<o<t�<�/<ě�=�P<49X<��<���<u<e`B<D��<�`B<�t�<���<�C�<��
=+>1'=�Q�=o<���<��<�/<�h=]/=<j=�P=,1=\)=P�`=\)=L��=�
==m�h=e`B=�+=0 �=#�
=@�>%=���=�%=P�`=ix�=H�9=q��=L��>�V=T��=���=ix�=�%=���=�^5=�`B=�-=Ƨ�B!�B	,bB��BЪB�B
��Bn�B'�~B�	B�B .	B"GA���B"��B�6BPB/�BY'B��B�eB�WB]jBk�B$wB�uBM-BV�B��Bn�B��B��BKB�jB!�@B��B~VBl�B�%B%<pBS>B�mBcBLzB KB�bB�B�Bd�B	��B4��B-U>BV�Bq�B	�B:LB!��A���B؉BpB?BEBo�B!ޞB��B�7B�+B�|B
�+BA�B&ߟB�NB��B ?�B"@�A�SgB"��B��BkB<QBAHB�@B��B��B��BBB FB�0B@>BvB�=B�B�+B>=B6B�B!��B��B@B��B?(B%?�B��B��B�B�B ?�BçB��B%dB19B
\aB4��B-?tBELBC�B	��BG:B!��A�B/B��BHOB@B>bBM�@�UAA�]�A���AF�$A��oA�EA@\�@��nAQҔAe��?@GK@�n�A�Q@���?�SA�pAGG�A�=�A��!AʭkBFlA��)A�(A��A��A��A�OC�u�An	LA�@�ƊAt��A_l A[A���A�&�A9�!@���A��@���A�wA�^�A4leAܢJA���A���A�R?A���A��gARt<AT�Beg@���AٚmA�m�A ��A��@B�A��A��IA��OB�t@�4]AƄ�A°�AG�OAգHA�~_A@��@�%BARdAc�?O�d@��
A��+@�Ĳ?��0A�׾AF�HA�A��TA�i�B�A��AʝA��A�78A͏:A��C�w�Al�;A��@��At�bA^�nA[	+A��FA��.A<��@��)A΀@�A��A� &A3PA��A���A���A�|�A�{oA�jAU AZBD�@��/Aٓ�A�mXA v2A��5@$�A��-A��A�n�B�{      +      B                               	                           s   H                                          K             	         Y   &            	         �            	         "   	   
            =            %      #                     #                  5                     #                        5                     %                        ;                                       %            %                                                                                          '                                                                        NT�OLQ�NmaO��NQ
`M�B'NgN�O�E;OHOO�{�M�;N��~N��WN��BNP)�N��O���N3�Nl�N!ZNQFHO
twO�O���Oy3N
�5OO�NF�N���O�s6O*��OZ�O��N��O��Nv{O �'O�9�O75�O1��O�AN��TN��)N�VVO��^OqO�vN�*SO_��N�әO!ަN��O��M��O"G<N%<�N�'OM��ORW�N���Ng��N���  �  P    �  }  �  �  O    �  \  z  �  F  {    �  i  �  @  �  �  
�  
;  �  �  �  ,  E  "  �    �  l  `  �  �  c  �  �  �  �  �  �  �  Z  Z  t  �  E    �  t  �  �  �  "  �  Q  q  �  U�#�
��o%   <�9X;��
;D��;��
;��
<t�<#�
;ě�<T��<u<o<t�<t�<e`B<D��<�C�<T��<�o<�o=�{<��<�1<���<�1<�1<�9X<�`B<�/<�`B<��<�`B=+<�<��=P�`=C�=+='�=\)=\)=t�=�%=,1=#�
=#�
=#�
='�=<j=<j>\)=H�9=Y�=Y�=m�h=�+=�\)=�9X=���=�9X��������������������HEEJN[gt����wtg[XNH��������������������OJMQ[ht�������th[QPO��������������������~}���������~~~~~~~~)3585)!/3<Un{������{nUI0*!)57<=@=5)&#/6<RZ^^UH</'#����������������������������������������"/;=>;3/"����������������������������������������/6BOS[\[OB;6////////ohp��������������|omdnz����znmmmmmmmmmm��������������������$)26BEEB?61)$$$$$$$$		$��������������������������������������������������������������� %#��hkttv����thhhhhhhhhh����������������������������������������15BN[gtttrgb[QNB;851�����
#)/645/#
���������		�����������  ������������
�����������������������)5ABIHB?5/))595/))��������������������%!$1CO[htvpsh^WMB6)%�
#0<CC><70#
�dbach�������xtqkmlid
#*//3561/#
'*+-/2<ABB><4/''''''93:<AHUY_VUHG<999999gmkmz���������zmgggg�������������������������������������� ���% %)45BNVNLFBA53)%%U]]`egt���������qg[U{}�����������������{��������������))11/)�������!'(&!����eegty|~tsgeeeeeeeeeey|yyz�������������zy��������������������eafjmz}�}zmeeeeeeee)16<BO[`bbOB6)��������������������������������������������������������
!#-/0/###
����� ���
��������������U�a�n�zÀÇÌÊÉÇ�z�n�a�U�Q�J�G�H�N�U�/�<�F�H�<�<�/�$�#��#�$�/�/�/�/�/�/�/�/�s�������;ԾҾž�������f�W�I�D�A�M�Z�s���)�6�:�6�)�(������������������� �������������������������������޾f�o�o�p�f�b�Z�V�R�S�Z�a�f�f�f�f�f�f�f�f�����������������������w�v�q�q�d�a�^�f���ʾ׾���� �����׾ʾ����������������ʿT�m�y�������m�h�G�;�.�"�����"�,�G�T�����	�����������������������������������������������~�x�v�x�|�������������T�a�j�m�q�q�o�m�a�\�T�P�K�L�T�T�T�T�T�T�����ʼ̼Լмʼ��������������������������'�0�3�;�9�3�'�"�����'�'�'�'�'�'�'�'�����������{�z�y�u�t�z����������������������������������������s�f�a�\�_�f�s����/�<�=�>�=�<�/�+�%�/�/�/�/�/�/�/�/�/�/�/����������������������������������������ÓÖàäåàÓËÇÂÇÐÓÓÓÓÓÓÓÓ�6�C�O�\�h�k�q�h�\�R�O�C�6�3�6�6�6�6�6�6�/�4�;�>�<�;�5�/�"���	�	�	����"�*�/ÇÓàïùþþõìàÓÇ�|�z�w�t�v�zÀÇ���(�5�A�N�Y�Z�U�N�A�5�(����	��	��A�N�Z�g�s�w�������s�j�g�Z�N�F�A�<�8�7�Aù������������ùøñùùùùùùùùùù����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ������������������y�p�m�g�`�^�`�m�y������)�2�7�7�5�8�5�)������������������_�l�u�v�x�|�{�x�l�_�S�F�>�@�F�H�R�S�Y�_���������Ŀſȿ��������������������������	��"�.�;�G�N�R�G�=�;�.�"��	������� �	�	���"�"�"�"����	��������������	�	ŭŹ����������������ŹŰŭŤŠřŠšŭŭ�����������������������������A�M�T�Z�\�]�Z�M�A�:�4�(�"�����(�4�A���л���-�7�;�6�'����ܻлû�������������������������ۼ׼ۼ��Ｄ�����ʼм˼������������r�e�f�r���������(�5�A�C�M�A�5�4�(������������s���������������������|�s�g�s�s�s�s�s�s���(�3�(�'� ���� ����������h�tāĄČčĎĎčā�t�p�h�c�]�a�h�h�h�h�������
���"�"��
���������������������������������������������������������������	��"�$�/�4�8�/�.�"��	������������������	���"�#�%�'�"���	���������������/�;�H�T�a�c�h�d�a�T�H�;�/�"���"�'�&�/�ʾ׾۾�����������׾ҾҾоʾľžʽl�y���������������{�y�p�l�f�`�^�\�`�e�l��������������������������������򼘼����ʼ׼����ּʼ������������������O�[�\�^�[�O�B�B�B�M�O�O�O�O�O�O�O�O�O�O���)�6�=�;�6�1�)�#�����������������������������������������������������ŠŭŹ��������ŹŭŠŜŗŠŠŠŠŠŠŠŠ�������������ɺ׺ֺȺ������������������������������������������������������������s�����������������������������s�g�d�f�s���(�4�-�(� ����	���������ǡǭǯǱǭǭǡǙǔǐǈǈǈǎǔǝǡǡǡǡ _ . - < Z w X ] E a G  H 7 @ | $ s 8 K � @   3 C V A _  L $ P [ + b K f . f 5 o M I 4  + T 1 o Y *  K 3 R C | E Y ; I  8  �  n  *  m  E  �  '  �  �  /  �  �  �  n  �  b    �  U  �  H  �    S  ;  �  ^  ^  �  �  @  Z  C  ,  �  7  >    �  @  �  �    #  �  A    �  �  �  �  �    n  4  �    �  ;  l  �  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  Cg  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  �  �    /  @  K  P  N  C  &  �  �  h    �  2  �  �  �   �  �  �  �            �  �  �  �  a  4    �  �  j  (  �  x  �  �  �  �  �  �  �  �  �  u  L  !  �  �  �    �  �  �  q  w  {  {  |  |  |  |  {  w  q  h  ^  G  *  �  �  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  i  V  B  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �           O  G  <  .         �  �       �  �  �  �  h  <    �  ,  �  �          �  �  �  �  �  �  n  F    �  �  �  I  �  o  }  �  �  w  f  O  /  �  �  �  �  �  �  M  �  �  9  �  f  \  S  J  B  ;  ;  :  :  2      �  �  �  �  �  �  t  \  D  ;  L  ]  m  w  y  t  e  T  >  #    �  �  T  �  �    b  �  �  �  !  B  \  o  |  �  �  �  �  s  Y  3  �  �  Q  �  ^  �  @  B  E  D  A  =  4  +         �  �  �  �  �  �  x  u  q  {  s  l  e  ]  T  J  A  6  (      �  �  �  �  �  �  �  �      	  
                                  �  �  �  �  �  �  �  �  �  �  �  �  r  Q  '  �  �  �  q  #  i  e  a  ]  e  p  {  r  ^  K  9  )      �  �  �  �  �  �  w  w  w  w  �  �  �  �  �  �  �  �  �  r  O  +    �  �  �  @  7  .  %          �  �  �  �  �  �  �  �  �  w  `  H  �  �  �  �  v  g  X  I  :  +    �  �  �  �  k  -   �   �   q  �  �  �  �  �  �  �  �  �  �  �  h  :    �  �  �  ~  U  )  �  �        	  	�  
6  
�  
�  
�  
�  
�  
�  	�  �  �  �     �  	q  	�  
'  
:  
8  
.  
  
  	�  	�  	�  	A  �  �    v  �    <  �  X  �  �  �  �  �  �  |  h  Q  6  "    �  �  t  3  �  �  r  �                                        �  �  �  �  �  �  �  �  �  �  v  Y  A  +  #  "      	    ,        �  �  �  �  �  �  �  �  �  �  �  v  g  Q  ;  &  E  ;  1  '      �  �  �  �  �  �  j  F     �  �  �  �  o             �  �  �  �  �  �  |  q  ^  >    �  �    :  �  �  �  �  �  �  �  }  ^  .  �  �  b    �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  z  M    �  �  c     �  �  �  �  �  �  �  �    a  :    �  �  �  �  �  �  �  x  U  l  e  ^  V  I  <  ,      �  �  �  �  �  z  _  D  '  	   �  A  I  U  _  V  A  (    �  �  w  4  �  �  J  �  �  /  �  (  �  �  �  �  �  p  ^  L  8      �  �  �  �  �  c  D  %    �  �  �  �  �  y  W  -     �  �  `  '  �  �  �  ]  4  �  �  7  I  Q  K  #  /  ^  H  '    �  �  �  l  C  �  \  �  ]   s  �  �  �  �  �  �  �  �  r  \  A    �  �  �  >  �  �  F    �  �  �  �  o  `  ^  `  d  ^  G  "  �  �  �  b     �    O  �  �  �  �  �  �  �  j  G    �  �  l    �  5  �  *  {  �  �  �  �  s  Z  B  )    �  �  �  �  o  ^  N  Z  q  �  �  �  �  �  �  �  �  �  �  u  e  V  F  7  &      �  �  �  �  �  �  �  �  �  �  x  ]  >    �  �  �  m  q  r  M  &  �  �  �  
'  
�  
�  
�  �  �  �  �  �  �  b    
�  
D  	�  �  �  �  �  �  C  R  Z  W  L  8  "    �  �  �  o  .  �  �  6  �  r  �  C    W  Y  U  H  /    �  �  {  :  �  �  \    �  �  F    �  t  r  n  b  Q  <    �  �  �  �  l  H  %    �  �  �  �  �  �  �  �  �  |  d  M  7      �  �  �  �  �  �  �  �  @  8  E  ;  1  .  /  ,      �  �  �  �  w  Z  =       �   �   �          �  �  �  �  �  |  _  B  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  c  R  A  6  -  %      �  �  �  �  K  �  )  e  r  \    �    I  >    �  s  	|  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  j  s  }  ~  ~  y  j  V  :    �  �  �  z  J    �  y  -  
  �  �  �  �  �  �  �  �  �  �  �  y  l  ^  Q  C  /      �  �                !    �  �  �  �  �  �  ^  6  "  1  ?  �  �  �  �  �  �  �  v  c  t  L  #  �  �  �  �  �  �  �  �  Q  M  7    �  �  �  �  |  U  %  �  �  �  m  /  �  �    F  &    �  (  K  h  d  <  
  �  �  N    �  ]    �  7  �    �  �  �  �  n  \  L  =  +      �  �  �  �  �  Y  /    �  U  7    �  �  �  �  �  h  G  %     �  �  �  P    �  �  L