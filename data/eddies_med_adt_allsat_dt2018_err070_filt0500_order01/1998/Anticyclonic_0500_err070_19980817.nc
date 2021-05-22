CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�9XbM�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N;   max       P��U      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @E�Q��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��Q�    max       @v���
=p     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q`           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @��          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >p��      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B-~�      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��5   max       B-�|      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?8t   max       C��`      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�v�   max       C�j�      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N;   max       P�t�      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?���n/      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >J      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���R   max       @E��Q�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v���
=p     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q`           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�]@          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E`   max         E`      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?���n/     �  V�               #   B   K   -   F   6            =         ^            T                     !      	   8                     $         
            (               `   <      �   9      
      $         
   	   %N��KO	RxN<M�O	I�O��PpZ�O��PO�'�P�V!P�XO�m[Oz��N�<�PC�^N��9PۃP��UO�Z\O�7N*?XP?�|N8��Nt�N?��N���N7F�Nb${O�C�N���N�*�P�FN� (N;Nh1N��N=*N�8fO�y&N<e�N-X�N���O�BON^s_NW�"O�dN��O_��N#��Og#O�y�O�mO��KO���O�"tO�G�N� YN,y�O N�|N��hN���NTO6P����ͼ�9X�T���D���49X�t���`B�D���o:�o;�o;�o<49X<e`B<u<�t�<���<�9X<�9X<�j<ě�<���<�/<�h<��=o=C�=C�=\)=\)=�P=�P=�P=�P=��=��=49X=49X=H�9=H�9=H�9=aG�=ix�=m�h=q��=u=�%=�o=��=��=��=�hs=��-=��T=��T=��=��=��=�{=��=ě�=���=���egkt�������tmjgeeeest��������������|zts;<<HPUX]UHD<;;;;;;;;�������������������������	
���������������#/8CD@AG/
��fccgnt�����������tkfswspsjt�����������ts�����)5FQVTB5���������	#/ajaYL971#�������������������������)9EFA6),./8<CHUY[UOH<3/,,,,�����
/<HUeihaH#
�����������������������pnlv��������������p����BNX\OIB5�����spqv��������������s������
#023/#
�������������������������)OZlst|mj[OB2fgitu}����tlgffffff#0<:40#����������������������������������������
')









����������������������������
��������kjhmz���|zzwtmkkkkkk}�����������������}}������)5BRRH)�����"),58<>755)"KN[gtutsg[[NKKKKKKKK����������������������������������������+*0<IKIC=<70++++++++����������������������������������������#/85/#!�����������������������������������������������'*%��������������������������9<HUWaaaaUOHA<999999�������������������������

��������#)6BEKMME:61
�����
��������������������
	6BO[g[VOB6)
#0<FIQRPIC90#[UU^m�����������zma[������

�����������������������������)*69DIE6)	�)5BCKKJB95)& WO[glnggg[WWWWWWWWWW��������������������������������������������������������
##&## 
�����������������������������

�����������������������������������������N�O�T�T�Q�T�N�G�I�B�5�.�*�.�1�3�5�B�M�NE\EiEiElEiE`E\EPEJEJEPEZE\E\E\E\E\E\E\E\���� �����������ܹԹѹܹݹ��軞���ûٻ׻λû��������x�p�l�s�x���������5�N�g���������������Z�5���������a�n�zÐàææØ�z�n�a�U�H�<�6�@�B�I�U�a���лܻ���4�M�F�4������ϻû��������<�I�b�{ŇŚŜŗŋ�n�U�#�
�������������<�/�;�T�m���������������m�T�;��	����/���(�A�Y�e�g�o�t�g�N�>�5����������	������	������������������������������������������������������������"�;�T�y�����~�r�`�C�.�"��������	�"���%�)�3�)�)���������������(�A�Z�f��������s�f�Z�A�(�����������	�������ƧƁ�g�N�.�.�O�uƎƧ���������(�.�2�(�����ݽ׽Ľ����Ľ˽�4�A�Z�b�n�s�r�f�^�Z�A�4�'�%�'�1�2�,�,�4���������������������������������������޾׿	� ���	���Ծʾ����w�{�������������{Ǉǈǉǈ�{�w�o�b�b�b�d�o�w�{�{�{�{�{�{�ּ��������ּѼμֻּּּּּּּּּ-�2�.�-�&�!���
��!�$�-�-�-�-�-�-�-�-�;�H�T�]�a�j�a�a�T�H�>�;�4�8�;�;�;�;�;�;�������������������������������������������
������
���������������������������������������������x�k�f�[�W�Y�f�l����������������������������������������׿������������ĿɿſĿ��������������������N�g�v�a�[�[�c�s�r�a�5���ܿϿ޿߿���N�������ĿɿĿ����������������������������������������{�y�v�y�y�������������������<�H�I�S�I�H�<�/�)�*�/�8�<�<�<�<�<�<�<�<�A�M�Z�c�d�[�Z�M�A�6�4�(�4�5�A�A�A�A�A�A���������������r�q�r�u��������E�E�E�E�E�F
F
E�E�E�E�E�E�E�E�E�E�E�E�E�FF$F1F=FVFcFoFxF�F�F|FoFVFJF=F1F#FFF�[�g�i�g�f�f�[�N�I�M�N�Z�[�[�[�[�[�[�[�[�T�`�j�f�`�T�G�D�G�K�T�T�T�T�T�T�T�T�T�TÇÓàìíôðìàÓÉÇÃÄÇÇÇÇÇÇ�S�`�l�������������y�l�`�X�Q�K�I�I�G�I�S�B�O�Q�Q�O�B�7�6�)�$�'�)�6�@�B�B�B�B�B�B�(�1�-�*�(�����
����#�(�(�(�(�(�(āčĚĦĳĽ��ĳĦĚā�{�t�[�O�H�G�O�hā¦ª¨©§¦ �e�r�~���������Ǻ̺ɺ��������~�v�r�f�a�eāĀ�t�n�h�d�h�tāćĄāāāāāāāāā������� �������������������������޼���������üż����������r�f�\�Y�\�c�r����ֺ�������������ֺɺ���������ŔŠŭŹ��������������ŹűūŧŠŘœőŔD�D�D�D�D�D�D�D�D�D�D�D�D�D�D}D�D�D�D�D��ûܻ�����ܻлû������������������û_�x�������~�y�x�c�_�S�F�:�0�,�2�:�F�M�_�"�#�.�*�#�"��	����������	��"�"�"�"�"�����������������������������������������o�{ǈǔǖǠǡǔǈ�{�o�b�a�a�b�g�o�o�o�o�
�
������
������������
�
�
�
�
�
�B�N�[�g�t�t�t�n�i�g�[�W�N�I�B�>�B�B�B�B�U�b�n�o�z�{�{�{�n�c�b�U�N�S�U�U�U�U�U�UE7ECEPE\EaE\EYEPECECE7E4E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEmEjEu . g H % - J F Q " J ] 9 & E ! $ ? * g D 1 S # j ) ( ; ( t N N C � 6 * 8 V X g 9 0 n i i V _ l a /  2 r 6 ? S G H $ a > 6 Z 8    �  �  X  +  R  �  �  }  �  �  �    �  x  �  �  ^  �  �  ?  7  f    �  �  O  �    �  �  _  �  �  q  �  ]  �  L  ^  >  �  *  z  �  �  �  7  f  P  �    �  Y  9       .  *  �  �  �  T  ����
�u�ě�;D��<�j=ix�=�C�='�=�7L=]/<�h<�C�<�=��P<�9X=<j=�`B=T��=@�<�`B=�"�<��=\)=t�=��=��=t�=�7L=��=0 �=�j=0 �=�w=P�`=m�h=49X=�o=���=]/=]/=m�h=�{=u=�%=ȴ9=�C�=���=���=���>#�
>   =��`>p��>I�=���=�j=� �=��=���=���=��=�"�>C�B	��Be�BT�B�&B#EBS�B
YBB�BvMB[EB�.B�VB�B�VB��Bg�BnB#�uB��B؊B	��B%tJB��B��B�JB.B"��A���B
��B{NB��B�B�B�|B&%Bk�B��B׈B�EB!�B-~�B ��BkBq�BN+B>�B��B��BqzB%��B DoB��B��B� B+�B	
�B� B�#Bv�B3/BM�B�^B	��BqB@B�B#7�B��B
@2B?�B��BŢB;�B9B�B�LB�BZ!BK4BH�B$)�B�CB�RB	��B%I�B ?�B�3B�nB�B"÷A��5B?{B� BȬB	?�B��B�fB&;�BE�B�BưB��B!�wB-�|B �
B��BęB�BD~BC�B�VBE�B%��BCkB:�B�B93B<B	6
B�^B��By�B=
B8�BôA��2A�@[C���?8t@���A� wA�q�@�3�A�kA��cA�gIA���A���AdӰA�*�A<{OBU�A.�A;'�A���AQ��B��A1S@k��A��Aq��A�rV@��"A���At��A�kEAu(�AoM�A�{LA<x@�Z�C�y�C��`A���Ag��A��Ap�A��A5�VA��A��C@ҕA���Aѵ�@��@C\BA�C��@�
�@��A�ekA��lBkIA�~�A�f3A��
C���C�	�A���A��C��!?NC@��A�|A�y�@�A� \A���A��1A���A�wKAgY�A�}KA;�B�<A.��A<�A��AR�B>A�@h{A���Aq�A�}�@髡A���Au�A���Au/JAm��A��A<�N@�-C�j�=�v�A���AgY�Aʊ�A�sAפ�A4�+A���A�z�@�xA܈�A���@��@K�A��KC��@��@�A��4A���B??A��VA��A��C���C�k               #   C   L   -   F   6            =         ^            T      	               "      	   8                     $         
            (   	            `   =       �   9            %            	   &                  3      +   5   9   %         1      '   ?   !   #      /                              ?                                 !                        !                                                         )      '   -   +            +         1                                          ?                                                                                                N��KN���N<M�N�lcN�%�Ph
O	O��PfZ�P!OLx_O`PnNy��P��N��9O��Pz\�O*4�O�.�N*?XO�\zN8��N"�
N?��N���N7F�Nb${OR��N���N�*�P�t�N� (N;NQ�N�cN=*N�8fO�y&N<e�N-X�N���OT�Ni�NW�"O�ǗN��O_��N#��Og#O�_oO�+�O��KO4��O��UO�G�N� YN,y�O N�|N��hN���NTO6P�  �  �    �  �    �  �  �  �    �  �  �  v  �  	  
  c  �  �    �  �  K  �  ]  �  �  
  �  �  �  $  U  �  u  �  �  �  S  }  q  �  �  �  u  �  j  G  	�  9  q  
  �    �  
h  �  \      	ż��ͼ��
�T���t�;�`B<D��<�j;D��<t�<�o<49X;��
<u<�j<u<�`B=#�
=+<�j<�j=]/<���<�h<�h<��=o=C�=�w=\)=\)=��=�P=�P=��=�w=��=49X=49X=H�9=H�9=H�9=�%=m�h=m�h=}�=u=�%=�o=��=�9X=�7L=�hs>J=�1=��T=��=��=��=�{=��=ě�=���=���egkt�������tmjgeeee��������������������;<<HPUX]UHD<;;;;;;;;������������������������������ ��������������
#(57430%
���olmst�����������utooswr{�����������xzzvs����)5?KOPMB5������/<HSWF3/*#
�����������������������)8>DD?6)//0<<<HTUHF<9///////�����
0DO_eaUH<#
����������������������{z}����������������{����)BLTRJB5)��{wy���������������{������
#022/#
�������������������������)6BOV]baYOB8)fgitu}����tlgffffff#05700#����������������������������������������
')









����������������������������
��������kjhmz���|zzwtmkkkkkk}�����������������}}������)5BQSG)�����"),58<>755)"KN[gtutsg[[NKKKKKKKK����������������������������������������+*0<IKIC=<70++++++++����������������������������������������#/85/#!������������������������������������������������ 
�������������������������9<HUWaaaaUOHA<999999�������������������������

��������#)6BEKMME:61
�����
��������������������)6BJOQOLB6)#0<EIPQOIB80#[UU^m�����������zma[�������

 ����������������������������)*69DIE6)	�)5BCKKJB95)& WO[glnggg[WWWWWWWWWW��������������������������������������������������������
##&## 
�����������������������������

�����������������������������������������5�B�N�P�N�M�N�R�N�B�@�5�2�0�2�5�5�5�5�5E\EiEiElEiE`E\EPEJEJEPEZE\E\E\E\E\E\E\E\���������������ܹ������軞�������ûĻû»�������������������������5�A�g�����������s�Z�5������������a�n�zÇÈÎÇÆ�z�n�l�a�]�U�P�N�U�V�a�a����4�@�C�@�4������ֻû������ûлܻ��<�I�b�xŏőŋ�~�n�U�<�#�
������������<�/�T�z���������������a�H�;�"�����#�/���(�5�A�N�R�]�^�Z�N�A�5�(����	�����	�����	������������������������������������ ��������������������������.�;�T�m�������y�`�G�.�"��	��
����.���%�)�3�)�)��������������4�A�M�T�f�s���z�s�Z�A�4�(�����)�4Ƴ���������������Ƴ�~�`�S�P�R�\ƎƳ����������������߽ݽͽͽнܽ�4�A�Z�a�m�r�q�e�\�M�A�4�(�&�(�2�3�.�/�4���������������������������������������޾ʾ׾�����������׾����������������{Ǉǈǉǈ�{�w�o�b�b�b�d�o�w�{�{�{�{�{�{�ּ�������ּռҼֻּּּּּּּּּ-�2�.�-�&�!���
��!�$�-�-�-�-�-�-�-�-�;�H�T�]�a�j�a�a�T�H�>�;�4�8�;�;�;�;�;�;�������������������������������������������
������
���������������������������������������������|�o�f�c�\�^�f�r����������������������������������������׿������������ĿɿſĿ��������������������N�g�t�`�[�[�q�p�_�5�(��޿ӿѿ߿�� ��N�������ĿɿĿ����������������������������������������{�y�v�y�y�������������������<�E�H�R�H�G�<�/�)�*�/�:�<�<�<�<�<�<�<�<�A�M�Z�b�c�Z�Z�M�A�8�4�+�4�6�A�A�A�A�A�A���������������r�q�r�u��������E�E�E�E�E�F
F
E�E�E�E�E�E�E�E�E�E�E�E�E�FF$F1F=FVFcFoFxF�F�F|FoFVFJF=F1F#FFF�[�g�i�g�f�f�[�N�I�M�N�Z�[�[�[�[�[�[�[�[�T�`�j�f�`�T�G�D�G�K�T�T�T�T�T�T�T�T�T�TÇÓàìíôðìàÓÉÇÃÄÇÇÇÇÇÇ�S�`�l�y�����������}�y�l�`�W�S�O�N�N�P�S�O�O�O�O�B�6�1�)�'�)�)�6�B�K�O�O�O�O�O�O�(�1�-�*�(�����
����#�(�(�(�(�(�(āčĚĦĳļľĳĦĚ�}�t�h�[�O�K�O�[�hā¦ª¨©§¦ �e�r�~���������Ǻ̺ɺ��������~�v�r�f�a�eāĀ�t�n�h�d�h�tāćĄāāāāāāāāā������� �������������������������޼������������������������r�e�a�c�l�r����ֺ�������������غɺ���������ŔŠŭŹ��������������ŹűūŧŠŘœőŔD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��ûлܻ�����ܻлû����������������û_�x�������~�y�x�c�_�S�F�:�0�,�2�:�F�M�_�"�#�.�*�#�"��	����������	��"�"�"�"�"�����������������������������������������o�{ǈǔǖǠǡǔǈ�{�o�b�a�a�b�g�o�o�o�o�
�
������
������������
�
�
�
�
�
�B�N�[�g�t�t�t�n�i�g�[�W�N�I�B�>�B�B�B�B�U�b�n�o�z�{�{�{�n�c�b�U�N�S�U�U�U�U�U�UE7ECEPE\EaE\EYEPECECE7E4E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EzEmEjEu . ^ H !  P $ P  W Z - 1 I ! $ A  e D & S ! j ) ( ; % t N K C � 8 % 8 V X g 9 0 ` | i T _ l a /  2 r ' ; S G H $ a > 6 Z 8    �  �  X  �  �  �    �  �    �  �  �  �  �  n  O  h  �  ?  �  f  8  �  �  O  �  �  �  �  z  �  �  a  �  ]  �  L  ^  >  �  �  �  �  k  �  7  f  P      �  x         .  *  �  �  �  T  �  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  E`  �  �  �  �  �  �  �  |  r  f  Z  M  D  =  6  .    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  w  L  !  �  �  �  g  5    ^  n  |  �  �  �  z  m  ]  F  (     �  �  [  )  �  �  '   �  7  M  Y  d  g  b  {  �  �  �  �  �  �  Z  /     �  v  S  �    b  �  �  �        �  �  �  F  �  �    �  -  �  )  :  	5  	�  
�    d  �  �  �  �  �  �  M  
�  
h  	�  �  �    5  p  �  �  �  �  �  �  s  9  �  G  P  3    �  �  
  �  M  �  F  W  b  �  �  �    u  s  o  Z  /  �  �  \  -    �  �  q  �  �     7  g  �  �  �  ~  Y  .      A  S  >    �  T  �   �  �  �  �  �        �  �  �  �  �  `  :    �  �  U  �   �  �  �  �  �  �  �  �  �  n  W  =    �  �  �  �  i  B     �  \  }  �  �  �  �  �  �  �  x  g  R  ?  ,      �      �  }  �  �  �  �  �  �  �  w  a  S  <  
  �  5  �    b  f  M  v  p  j  c  Z  P  D  8  ,    �  �  �  �  �  w  e  V  H  :  F  Y  j  x  �  �  �  �  �  �  v  [  <    �  �  �  D  �  ;  y  �  �  �  	  	  �  �  �  b    �  /  �  "  �  �  Q  q  �     t  �  �  �  �    	  �  �  �  �  �  \  6  
  �  �  �     [  c  `  U  A  .  5  9  $    �  �  �  |  ?  �  �  [  �  (  �  �                            �  �  �  �  �  �  �  B  �  �  �  �  �  �  �  �  }  I    �    `  �  3   �    �  �  �  �  �  �  t  T  3    �  �  �    W  /    �  �  �  �  �  �  �  �  �  �  �    r  `  L  1    �  �  |  7   �  �  �  �  �  �  �  z  b  J  2       �  �  �  s  P  ,  	  �  K  A  6  ,  !    
  �  �  �  �  �  �  �  x  Y  :     �   �  �  �  �  �  z  n  b  T  D  4    �  �  �  �  �  ^  :     �  ]  X  S  N  I  D  ?  :  5  0  %      �  �  �  �  �  �  �  {  �  �  �  �  �  s  ]  F  (    �  �  g    �  �  K     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  E  0      
       �  �  �  �  �  �  z  _  @  "    �  �  �  w  E    �  �  �  �  �  l  >    �  �  l     �  �  Y    �    r  �  �  �  �  �  �  �  �  �  �  �    t  i  U  /  
  �  �  B   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      )  0  2  3  .  $      �  �  �  �  8  �  ;  �  �  /  S  U  P  J  ?  0      �  �  �  b    �  �  :  �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     u  d  [  `  f  j  d  T  <       �  �  �    _  ;      �  �  �  �  �  �  �  r  Q  *  �  �  �  /  �  b    �  �  F  y  �  �  �  �  �  �  �  �  �  e  I  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  O  4    S  9    	  �  �  �  �  �  �  x  ]  C  )    �  �  �  �  �  k  o  o  [  b  }  p  U  0    �  �  j  #  �  �  T  	  �  �  l  m  n  n  o  p  q  h  Z  L  >  0  "    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  s  �  �  �  �  �  �  �  �  |  C  �  �  @  �    ^  �  m    N  �  �  �  y  c  N  <  *    	  �  �  �  �  �  �  �  �  �  �  u  e  K  )    �  �  �  b  3    �  �  i  $  �  �  e    �  �  �  �  {  Y  7    �  �  �  T    �  �  ;  �  �  ]    �  j  e  a  \  V  K  <  (      �  �  �  �  �  {  _  ?    �  :  �    9  E  C  ,    �  �  ?  
�  
�  
  	u  �  �  �  7  �  	�  	�  	�  	�  	�  	p  	M  	1  	  �  �  X  �  �    c  �  !  �  �  9    �  �  n  1  �  �    M    �  �  �  X    �  0  u  �  2  �  �    N  i  q  b  D    �  S  �  �  2  �  #  !  �  
k  	�  
  
	  	�  	�  	�  	�  	R  	  �  �  K  �  �    b  �  �    \  �  �  �  x  W  7    �  �  �  �  ]  .  �  �  �  c  f  o  a              �  �  �  �  �  �  �  }  c  H  $  �  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
h  
=  
  	�  	�  	r  	5  �  �  x  /  �  �  ,  �  .  f  q  f  L  �  �  �  �  �  ^  1     �  �  N  	  �  q  (  �  �  E  �  �  \  K  :  (      �  �  �  �  �  �  �  t  ^  F  +    �  �    �  �  �  �  �  s  ^  G  /    �  �  �  �  c  9    �  �        �  �  �  y  O  &  �  �  �  y  K    �  �  �  J    	�  	�  	�  	�  	y  	K  	  �  �  2  �  {    �  E  �  ?  �  x  �