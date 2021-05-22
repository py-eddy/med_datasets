CDF       
      obs    N   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���
=p�     8  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�'    max       P��R     8  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��1   max       =+     8      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�(�\   max       @F��z�H     0  !T   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vt��
=p     0  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q`           �  9�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�I�         8  :P   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��G�   max       <�`B     8  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�l   max       B1��     8  <�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��-   max       B1�p     8  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >[��   max       C�WR     8  ?0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�x�   max       C�V�     8  @h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          B     8  A�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     8  B�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     8  D   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�'    max       P��'     8  EH   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?ІYJ���     8  F�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��1   max       =+     8  G�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�G�z�   max       @Fe�Q�     0  H�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @vp�\)     0  U    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  aP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�^        max       @�L�         8  a�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     8  c$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��1&�y   max       ?�|����?     `  d\         	      	                  .                           7   
   (      	   %                >                        /            "               !   !   
   	            %                     -            A   )   9            ,                     Nq�gN�ȠN��Nq��N�siOL<�OHĿN�A�OD4�O1	P�yO->QM�MUN*OC�iO>��M�' OQu�O��*P��RN��=O��N��PN�9�OX�NM��Nv�EN2�rO�YP,b�N�.�Nq�N�JO�]�Ok�qNl.O8��PN׍N���P�hO���O��2N��#N��}N�K�O$O���O,{N�R�N�:�N��7N5�QO��P��Nl�2O˖�N��M̷�O��*N�i�P��N��'OA� N�L�O�W�O�lO���N{�Ob�ZNn��O�I�OƌOo�N�pO@=�Nwj�Oo��N=�1=+<�<�C�<u<D��<49X<#�
;��
;��
;�o�o�t��t��#�
�49X�D���e`B��o��o��o��t����㼣�
��1��9X�ě����ͼ��ͼ��ͼ�����/��/��h��h��h��h��h��h���������o�o�o�+�+�+�C��C��\)��P����w��w�#�
�#�
�#�
�''',1�0 Ž0 Ž8Q�<j�D���D���]/�]/�m�h�m�h�q���y�#�����O߽�t���1% ������������������
#%$#
�����rtv��������wtrrrrrrr<<BIU_bntxnncbUIB<<<>BDNX[]gntuskg[NG@>>BHQUaiz����znaUH?>Btz�����������zzstttt@CO\hu|�����uhOIC>>@����
#
��������t������������volt`agmwz{��������zyme`LO[[__[XOJLLLLLLLLLL #*/;<=<9/&#        ����������������������������������������nnwz����znnnnnnnnnnn������������������������������#0Vn������{bU0)**,.14)(
�����
(,-+#
��������������������������hmvz���~|zmihfghhhh#),6BOU[ab`[ZSOB6(#jmz{z{zupmlliijjjjjj������������������������������������������������������"*<HUahpsrlaUOG& !"V[_gmtwxtpg[XSVVVVVV��������������������Yahmz����znymaaa_\YY����(������)5BIN[gt�����tg[NB5)MOT[hqttth[QOKMMMMMMlv����������������vl������0bfbUI0#����)5;ABBB5)$gm����������������gg_gt������������tga\_������������������������������������������������������������`anz������znkfaa````��������������������#/<NZjlaH</##+/8<AEFFC<1/###��������������������266>BCO[ab[YOOIB;622����������������������������������������9O[ht���th[OJMTOCA;91<Ha�������zna\QC<51��������������������S[g����������wtg^]RS�������������������������������������������
#0:ACB@<0#
���:<FIMUU[_a]UID@=<;<:����
������������������������������7=BN[_egjlhg[NB:8547�� 
#/,%#
��������������������������������������������)5BN[gdUKB5)
��������������������5BN[bgjtskg[NB>85335w�����������wwwwwwww����$*'������RYbn{���������{lWJJR������$�����������������������������������������������������������),8BO[ba^[OB6)#!##)8<DHILJH<99888888888�3�2�,�3�3�@�L�M�R�O�L�@�3�3�3�3�3�3�3�3¦¥¦§²¿����������¿²¦¦¦¦�����������������������ȼʼͼʼ������������ݿڿݿ�������������������ʼɼ����������������ɼʼּ׼��ּʼʼʿ������|�y�s�y�����������Ŀʿȿ����������A�;�4�3�2�4�>�A�E�M�Z�f�q�i�i�k�f�Z�M�A���������������������������������������Ҿ����������������žʾӾ׾׾ؾھ׾ξ�����ÓÏÓÝãàÜàìù������������ùìàÓ�s�k�j�^�U�Y�g�������������������������s�����������������������$�(�(�$�$����ÓÍÓÓàìîìåàÓÓÓÓÓÓÓÓÓÓ�H�?�<�5�<�>�H�L�U�X�U�Q�H�H�H�H�H�H�H�H�����������z�|������������žɾȾξоʾ��
�����������
��#�/�<�C�H�R�H�<�/�#��
�U�U�H�G�H�H�U�W�Z�U�U�U�U�U�U�U�U�U�U�U�{�v�r�p�v�{ŀŇŔŖŠŭŻŹŭŨŠŔŇ�{�	���������� �	��.�;�G�R�W�J�;�.�"�	���o�`�\�Z�W�_�s�������������������������x�v�l�e�j�l�x�����������������������x�x���{�w�������������Ŀѿ�����ѿĿ������m�h�m�u�t�w�z���������������������z�z�màØÓÍÓàìù������������ùìàààà�f�_�Y�J�=�<�@�E�M�Y�f�|�����������r�f�������������)�6�)���������������ŭũŧūŭŹ������������Źűŭŭŭŭŭŭ�n�l�b�a�_�a�n�v�zÃ�z�z�n�n�n�n�n�n�n�nÓ�n�m�e�[�W�\�a�n�{ÇÓìùþùôìàÓ�(������(�A�Z�s�������������s�Z�A�(�6�,�*�'�*�5�6�C�F�O�T�Q�O�C�6�6�6�6�6�6�N�K�B�5�3�/�5�A�B�C�N�O�V�[�[�[�N�N�N�N��������������������	���"�"�"��	�������׾ϾϾ׾����	����.�9�?�<�.���	������������	��� ������	���������������ûĻϻ̻û��������������������������x�r�x�����������������������:�!����պɺ��)�-�A�Z�f�Z�W�_�o���x�:��������������������������������������������ĸķĲĬĽ������������#�*������{�r�j�I�7�1�8�<�I�b�{ŇŠŦůũřŔŇ�{�h�O�@�?�B�F�O�U�[�hāčĚġĥĚĐā�t�hƎƅƁ�u�u�h�g�h�m�uƀƁƎƚƛƛƚƎƒƎ�h�a�\�O�C�;�6�4�6�9�C�O�P�\�d�h�j�k�h�h�z�w�p�s�z�����������������������z�z�z�z�/�$�"���"�/�;�H�T�Z�[�T�Q�H�;�/�/�/�/�����������ʾ׾�	�����	�����׾ʾ�E�E�E�E�E�E�E�E�E�E�E�E�F	E�E�E�E�E�E�E������������������������������������������;�/�.�"���
���"�,�.�;�;�G�R�G�C�;�;������������"����������������D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D߽Ľ��������ݾ������������ݽнĹ��z�w�y�����ùܹ���$������Ϲ��������~�}�r�e�]�Y�U�V�Y�Z�e�r�~�����~�~�~�~�����������������5�=�N�U�Y�N�B�)��*�'�%�*�6�;�C�O�T�S�O�M�C�6�*�*�*�*�*�*�4�)�'��'�4�:�@�M�P�M�@�4�4�4�4�4�4�4�4�ݽĽ��������������Ľнݽ������޽ݽ�����������(�4�A�H�H�A�4�(�%�����r�������ּ���	� ���	�����ּʼ����r�b�a�V�O�L�V�b�o�x�{�|�{�o�c�b�b�b�b�b�b�B�6�,�)�/�6�B�G�O�[�h�h�o�s�u�t�h�`�[�B����������������������������x�U�S�L�_�r�����������ûлܻ�޻������x�~�p�k�m�x���������ֺ�������ֺɺ��~ùòãÙÐÌ×éù����������������ù������r�q�r�x��������������������������#����� �#�/�<�H�K�V�`�^�U�H�B�<�/�#�<�6�/�4�<�H�U�X�V�U�H�B�<�<�<�<�<�<�<�<�:�.�!��޼׼ۼ����6�G�Z�}���y�`�S�G�:��ܻ׻ӻֻ�����'�@�K�Q�O�E�4���������߽��������(�4�6�8�2�(�����àÖÓÇÁÇÓàìóùúùìààààààĿĽĳĦĚčā�{�{āčĚĦĮĳĶĺ����Ŀ�����������������������������������N�Z�g�p�������������������������s�g�Z�ND�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� - A / / ` $ F / e L ? < N ! K a { 5 " ' ] O h l * � = Q L V  T \ M X V N a L ' G 2 2 Z E , > R 3 c V g C ] f I ? � @ k T , $ C H T < b  S ^ F L k X g R n  {  �  �  k    �  �  �  �  t    �    =  �  �  ?  �  o  ,  �  "  O  "  �  �  �  Z  �  T  �  �  �  �    �  �  �  �  B          �  +  �  �  �  �  �  �  C     �  �  �  .  �  7  �  �  �    _  y  @  Y  �  �  �  �    �  �  �  6  q<�`B<�j<o<49X;D���o�o�o�#�
��`B�]/�ě��49X�u�����C���C��#�
�#�
��hs��`B�m�h���ͼ�h�m�h����`B��w�ixս�-���+�o�Y��D����w�8Q콙���'q���y�#��o����w�\)�H�9��+����49X�,1��w�Y��e`B����0 Ž}�0 Ž49X��7L�ixս� Ž@������\)��G��� Ž����aG���������������-���������P�����BCpB�_B$~B
?�B'd5B��BF�BڻB1��B7�B0�A���B��B:]B!XVB�BW�B�<B�7B'�B�uB�B��A�DBU�A�� BQvBXzBV�BV�B	(PBm�A�lB�[B	<BjzBq�B$ŃB��B B
��B��B�BBBZ=B�tBPlB�uB��B-�B�lB�B��B }�B
cB?.B!�B$� B&�YB-/B}�BI�B#�By�BSTB�BE�B[�B
��B�cB(�B�rB%�B��Bt�BcB��B@B�?B$d9B
?3B'5�B��B�zB��B1�pB(
B�gB  �B1TB4�B!�B<
BIB��B��B&�nB�`B�B�MA� B?�A�n�B|�B��B��BJ4B	9cBL�A��-B�$B	pB��BrgB#��B��B
��B
�\B�>B�B=�B(wB@�B��B?�B�|BQ�B=�BBB��B@B @B:wB�B �XB$�9B&��B-@MB�\B��B�B_KB�	B��BB�B7B
�*B=]B)<B�CB
ΓB��B6�B�PBz?��GA��u@�CA�@�As<_A=��A�f�AN�GA͠,A�*B��A�_(A�kAL*A���AŎCA�8�A^^�A��D@�O�Av�UA�o�A�=@�S�A�H�A��[Aǩ{A��A�?UB ��A�A��AZ�`AZZv@�kk@�V�@m̰A�nA�h�A�6A�;
BW�BF�A��hA���AT��C�WRA�_<A`�eA���C�8#A+�>[��?��A��#B �@�ΰA&a�A6T�A �?BM�Aٲ�A��@��y@*$�A�A.@��vA�
�A��A
^�@���A2��Aː�AߴB�A��`C��Q?���A��@��yA�G5@���AsåA<�=A�!CAN�{A�:+A�}hB��A˃6AĎ�AJ�A��UA�l�A�A^|A���@�UAv��A�|�A���@�AӴVA���Aǀ�A�}[A��GB �%A���A�܉A[��AZ�u@���@�/@dd�A��CA��A�A�{�B�~B1�A�4�A� �AW �C�V�A���Aa	LA�sC�?�A*�=�x�@vnA���B 5`@��A%�A5#�A!�BI�A���A�@@���@R�A�<@���A �A���A	$@���A2��A�stA߀B��A��xC��6         
      
                  /                           8      (      	   %            !   >                        /             #               "   !      	            &                     .            B   )   9            ,                                                       1                           5      %                     !   -                        9      )   #                  !                  !   -      !               1            +   '   %            +   #                                                   '                           5                              #                        '                           !                     -                     )            '   '   !               #                  Nq�gN�ȠN��Nq��N�wFOټOj�N�A�N��N��mPr?N��M�MUN*O+�Of�M�' O@�OO|�P��'N��=O��aN��PN�9�Oz�N$��Nv�EN2�rO�õOڵ�Nq^Nq�NA��O��O)��Nl.O
��O��KN���O�) On?%O]�*N��#N��N�K�N֙�O���N��N�	�N�:�N��7N5�QO7OP��Nl�2O��LN��M̷�O�D$N��PH�N��'O2]�N�L�O�pO�lO�'�N{�OW`�NI�O5OƌO]�N�pO(olNwj�OPʑN=�1  �  �  r  �  �  A  L  _    �  L    �  <  3    �    c  :  �  �  f  �     �  �    "    �  �  q  �  �  �  �  D  �  K  *  N  �  �  �  �      �  �  �  �  �  �  N  U  3  �  s  �  �  �  �  �  
  �  (  
  �  �  &    "  �  P    ~  P=+<�<�C�<u<#�
;�`B;�`B;��
$�  $�  ��9X�D���t��#�
�D����o�e`B��1�ě���C���t���`B���
��1�����ͼ��ͼ��ͼ�h�'�`B��/�����+��h�o�8Q�����#�
�\)�o�+�o�t���P��w�\)�C��\)��P�0 Ž�w��w�49X�#�
�#�
�0 Ž0 Ž,1�,1�49X�0 Ž@��<j�e`B�D���aG��aG������m�h�u�y�#��7L��O߽�����1% ������������������
#%$#
�����rtv��������wtrrrrrrrHIUbnpnjb]UIE?HHHHHHABJN[egpqojg[NKDBAAAEHUXadnx��{zynaUHDBEtz�����������zzsttttCCO\hru}uh\OHCBCCCC���

���������������������������}xvw�fmmz����������zpmhffLO[[__[XOJLLLLLLLLLL #*/;<=<9/&#        ����������������������������������������nnwz����znnnnnnnnnnn�������������������������������	#0n������{bU0
	)**,.14)(
������
!
���������������������������hmvz���~|zmihfghhhh')*068BO[]^\[XSOB63'kmzzzzzqnmmljikkkkkk�����������������������������������������������
��������,3<HU^elnlfaU<1*%)),V[bgkssng[ZTVVVVVVVV��������������������[ajmz~zymila`][[[[[[������������GNP[gt{�����tg[NNBGGMOT[hqttth[QOKMMMMMMpty��������������{tp����
0BQQI<#
������)5;ABBB5)$nw��������������xqlnnt{�������������tjgn������������������������������������������������������������`anz������znkfaa````��������������������#-<K]cfbUH</& #&//0<<ACB></#"  ��������������������266>BCO[ab[YOOIB;622����������������������������������������GOR[hrtw|~}ztih]TOKG1<Ha�������zna\QC<51��������������������[qt~����������tg_[X[�������������������������������������������
#08?@?90#
����<<=AIKSUY]_ZUIFB><<<��������	��������������������������568?BN[_dgikgg[NB955�� 
#/,%#
��������������������������������������������)5BN]\PLFB5)��������������������356BN[agsrjg[NB>9543{�����������{{{{{{{{���  ����RYbn{���������{lWJJR�����!#���������������������������������������������������������%)6BOY[`_[OB62)($"!%8<DHILJH<99888888888�3�2�,�3�3�@�L�M�R�O�L�@�3�3�3�3�3�3�3�3¦¥¦§²¿����������¿²¦¦¦¦�����������������������ȼʼͼʼ������������ݿڿݿ������������������꼽�������������ʼӼּܼݼּʼ�����������������������������������ÿ��������������A�=�5�4�4�4�;�A�M�Z�f�g�f�d�e�f�_�Z�M�A���������������������������������������Ҿ����������������ʾ;ϾѾӾʾ�����������ìáæèìù��������������ùìììììì���{�t�i�_�g�s��������������������������������������������$�%�%�$� �������ÓÍÓÓàìîìåàÓÓÓÓÓÓÓÓÓÓ�H�?�<�5�<�>�H�L�U�X�U�Q�H�H�H�H�H�H�H�H������������������������þǾǾʾ̾ξʾ��
������������
���#�)�/�:�<�=�/�#��
�U�U�H�G�H�H�U�W�Z�U�U�U�U�U�U�U�U�U�U�UŇł�{�w�s�y�{ŇňŔŠŤŭŭũŤŠŕŔŇ�	���������	���"�.�7�;�E�G�;�.�"��	�����p�a�]�Y�a�s�������������������������x�v�l�e�j�l�x�����������������������x�x�������������������Ŀݿ��ٿѿĿ��������m�h�m�u�t�w�z���������������������z�z�màØÓÍÓàìù������������ùìàààà�r�j�f�Y�P�M�B�A�K�M�Y�f�i�r���������r�������������)�-�)���������������ŭũŧūŭŹ������������Źűŭŭŭŭŭŭ�n�l�b�a�_�a�n�v�zÃ�z�z�n�n�n�n�n�n�n�nÓÇ��r�i�^�Z�a�n�zÇÓìùüùðìàÓ�(�����(�5�N�Z�s�����������s�Z�A�5�(�6�-�*�(�*�6�C�O�S�P�O�C�6�6�6�6�6�6�6�6�N�K�B�5�3�/�5�A�B�C�N�O�V�[�[�[�N�N�N�N������������������	����	�����������������׾оо׾���	���.�8�;�=�9�.����������������	��������	�������������������ûĻϻ̻û������������������������������������������������������������������!�-�:�D�H�D�K�S�W�:�-�!�����������������������������������������������ľĽĽ��������������������������{�w�n�d�P�M�U�b�n�{ŇŔŠŢťŨŢŔŇ�{�t�h�O�E�B�M�O�Y�[�h�tāčĚěğęčā�tƎƅƁ�u�u�h�g�h�m�uƀƁƎƚƛƛƚƎƒƎ�C�=�6�5�6�@�C�I�O�\�b�h�h�h�\�O�C�C�C�C�z�w�p�s�z�����������������������z�z�z�z�/�*�"��"�"�/�;�H�T�X�Y�T�N�H�;�/�/�/�/�����������ʾ����	�����	����׾ʾ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������;�/�.�"���
���"�,�.�;�;�G�R�G�C�;�;������������"����������������D�D�D�D�D�D�D�D�ED�D�D�D�D�D�D�D�D�D�D߽нɽĽ����Ľǽнݽ����	�������ݽй��z�w�y�����ùܹ���$������Ϲ��������~�}�r�e�]�Y�U�V�Y�Z�e�r�~�����~�~�~�~�����������������)�6�A�C�@�5�'���*�'�%�*�6�;�C�O�T�S�O�M�C�6�*�*�*�*�*�*�4�)�'��'�4�:�@�M�P�M�@�4�4�4�4�4�4�4�4�Ľ����������������Ľнݽ�����ݽнľ����������(�4�A�F�E�A�4�(����ʼ��������������ּ����������ּ��b�a�V�O�L�V�b�o�x�{�|�{�o�c�b�b�b�b�b�b�Z�O�B�6�0�+�0�6�B�I�O�[�h�h�o�r�s�h�[�Z����������������������������x�o�X�T�_�s�����������ûлڻ�ۻ������x�~�p�k�m�x���������ֺ�������ֺɺ��~øèßÚÙâù������������	���������ø������r�q�r�x��������������������������/�$�#�����"�/�<�H�J�U�_�]�U�H�A�<�/�<�8�0�5�<�H�U�U�U�T�H�@�<�<�<�<�<�<�<�<����������������!�.�0�5�/�.�!���ܻ׻ӻֻ�����'�@�K�Q�O�E�4�����������������(�5�7�1�&�����àÖÓÇÁÇÓàìóùúùìààààààĚčĂā�|�|āĄčĚĦħĭĳĵĹĿĳĦĚ�����������������������������������g�[�g�r�����������������������������s�gD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� - A / / a  J / Z : > 8 N ! M O { *  % ] U h l ' � = Q M ? # T q N C V 8 ] L , 9 0 2 C E ) = D / c V g @ ] f M ? � 2 U N ,  C C T 7 b  D ? F F k C g M n  {  �  �  k  �  >  _  �  +  �  �  !    =  �  N  ?  @  �  -  �  x  O  "  G  �  �  Z  v  �  q  �  ~  W  �  �  F    �  4  �  �    �  �  �  �    �  �  �  �  \     �  \  �  .  6  �  �  �  r    4  y  �  Y  �  �  4  �  �  �  t  �  �  q  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  w  h  Z  N  C  8  0  -  )  &  $  "  !        	    �  v  h  Z  L  =  *    �  �  �  �  �  �    i  T  ;  "    r  o  l  b  X  L  ?  1  "      �  �  �  �  �  �  t  |  �  �  �  �  �  �  �  �  �  �  �  �  u  f  X  I  @  9  1  *  #  �  �  �  �  �  �  �  �  �  }  h  T  A  *    �  �  �  1   �  
  )  5  <  A  @  5  #  	  �  �  �  �  \  -  �  �  k    �  $  4  D  K  K  C  9  ,        �  �  �  h  ;    �  �  �  _  [  X  T  O  K  F  A  =  7  1  +  "      �  �  �  �  �  �  �  �  �             �  �  �  �  �  X  0    �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  �    c  Y  U    "  ;  C  F  J  K  J  B  .    �  �  k    �  ?  �  �  Q  �            	  �  �  �  �  �  p  J     �  �  M  �  j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  <  ;  9  8  4  &    
  �  �  �  �  �  �  �    l  Y  E  2    .  2  0  -  !      �  �  �  �  �  r  5  �  �  x  3   �  �  �  �     �  �  �  �  �  ~  E  �  �  C  �  o  �  9  x   �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  s  �       
             �  �  �  �  p  @    �  �  o  J  �  �    9  N  [  b  b  Y  H  4    �  �  �  <  �  �  ,  �  3  2  4  3  0  !    �  �  �  G  �  �  G  �  �  v  (  �   �  �  �  �  �  �  �  �  �  �    e  F     �  �  �  U    �  �  9  b  w  �  �  �  �  z  e  H    �  �  T    �  P  �  O  �  f  ^  V  N  I  O  U  Z  Y  N  C  7  *      �  �  �  �  �  �  �  �  �  �  �  t  g  ^  V  O  I  B  6  +      �  �  �  �  �  �           �  �  �  |  H    �  o  �  {    �  �  �  �  �  �  �                  �  �  �  �  �  �  �  �  ~  r  f  Y  M  @  4  (        �  �  �  �  �  �  �  �        	    �  �  �  �  �  �  �  �  �  �  y  �  �  �  �  	    !      �  �  �  k  "  �  �  9  �  �  S  /    �  B  t  �  �  �      	  �  �  �  �  J  	  �  o    �  �  -  C  �  �  �  �  �  �  �  �  �  �  �  m  Y    �  �  I    �  {  �  �  �  �  �  �  �  �  �  x  l  ^  P  C  7  +        �  \  _  b  f  i  m  p  t  w  z  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    Z  4    �  �  �  �  K  �  �  "  ;  u  ~  �  �  �    t  g  S  4    �  �  A  �  �  K  �  3  �  �  �  �  �  �  �  �  �  �  �  �  m  H  !  �  �  �  v  H  �  �  �  �  �  �  �  �  �  �  v  a  I  *     �  �  [    �  .    �  �  �  !  <  @  '    �  �  �  T    �  r    �  �  �  �  �  �  �  �  �  �  z  b  H  )    �  A    �  �  V  3  �  �  �  %  =  I  H  7    �  �  �  �  �  �  ]    �  6  �  �  �  �  �    %  *  (    
  �  �  �  e    �  Z  �  -  %  �    B  M  E  .    �  �  j  1  �  �  �  P  �  �    j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Y  B    �  �  �  W     �   �  �  �  �  �  �  �  �  �  �  y  m  b  V  H  8  '       �   �  �  �  �  �  �  �  �  �  �  �  o  I    �  �  \    �  O  �  �         �  �  �  �  Q    �  �  >  �  �  '  �  �  �  ]  �  �  �        �  �  �  �  z  *  �  )  �  �    R  m  :  �  �  �  �  �  �  �  \  3    �  �  v  H  '    �  �  �  �  �  �  �  y  o  f  \  S  K  C  0    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  X  "  �  �  }  B    �  �  W    �  �  e  #  Q  g  {  �  �  �  �  |  v  l  ^  J  2    �  �  �  g  �  �  �  �  �  �  �  �  �  g  >    �  �  �  C  �  I  �     �  N  C  8  -  "        �  �  �  �  �  �  �  �  �  }  l  [  R  S  Q  T  J  4      �  �  �  �  �  h  7    �  �  i    3  )          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  n  �  �    <  j  �  �  `  n  n  c  V  E  1    �  �  �  �  _  !  �  �  r  -  �  @  j  i  �  �  {  g  N  2    �  �  �  }  M    �  �  G  �    O  r  E    �  �  �  �  g  8    �  �  5  �  �  ,  �  i  
  �  �  �  }  p  _  O  ?  .      �  �  �  �  �  ~  S  )   �  �  �  �  �  {  e  J  .    �  �  �  �  n  M  +    �  �  6  �  �  �  �  {  _  @    �  �  �  O  
  �  �  :  �  �  #  �  	�  
  
   	�  	�  	�  	[  	$  �  �  .  �  T  �  S  �  �  	  �  ~  �  u  P  *    �  �  �  z  M    �  �  �  E  �  �    \  �  �    #  (                �  �  D  �  p  �    �  �  
  �  �  �  �  �  �  �  �  �  �  �  x  c  O  :  &    �  �  �  �  �  �  �  �  �  f  4  �  �  �  L    �  �  c    t  �  l    �  �  �  �  �  |  r  h  ]  R  E  5  %      �  �  �  o  �  �  �  �  �  �      %  #      �  �  [  �  h  �         �  �  �  �  �  �  y  c  M  1    �  �  O  
  �  c   �    !  !      �  �  �  �  �  f  ?    �  �  �  b  b    �  �  {  f  K  1    �  �  �  �  b  >    �  �  �  �  �  �  �    :  A  '    �  �  �  n  :    �  �  7  �  l  �  �  �   �        �  �  �  �  �  �  i  F  $    �  �  �  �  c  7    .  l  }  w  h  Q  8      �  �  �  m  -  �  �  [    �  �  P  3      �  �  �  �  �  ~  r  g  W  C  1       �  �  �