CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�bM���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��w   max       P��]      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E������     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vh�\)     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @R@           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @�x`          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >�C�      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�u�   max       B,
      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�S   max       B,Z      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�P:   max       C�[      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��>   max       C�p      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         }      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��w   max       P|      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���?   max       ?��f�A�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       >L��      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�G�z�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vh�\)     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @R@           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�,        max       @���          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?�����m     �  V�                                        
                    }      
   	            (         8                        +   �   /                  3   ;   5         
      k            #               N   4M��wNfO9i�O=aN<�N�٦OWYN�&N�>�O�RN��EN��IN�ȰN�$�N��Oޠ�N9�O�kOs%�P��]N��	N|��N��~P�5Oec�N<
mO��NP�mN��aP*�IOS�dN�s1N��M���O;"O�P�9O�x�O���O�'O�x�O��N�"�M�$�N$�O�<Pe�O�UN���N���N��N�zlPPO�{�N���O'O3�1NE�{NX�N�",N�X�O�ݲOf���+���
%@  ;o;�o<o<o<t�<t�<t�<#�
<#�
<49X<e`B<e`B<u<�C�<�C�<�C�<�C�<�C�<�C�<�C�<���<���<��
<�9X<���<���<���<�=o=o=\)=t�=t�=t�=�P=��=,1=,1=49X=49X=8Q�=D��=H�9=L��=L��=P�`=P�`=aG�=e`B=ix�=m�h=y�#=y�#=�o=�o=��P=���=��=�v�=���:5:<ILNI@<::::::::::;:9<@IJKKIF<;;;;;;;;wz{��������������{zw��������������������dbfgitx~{tmgddddddddcehot��������xthcccc��������������������JGNS[gt|����|tqg[NJJMGIJNNP[agigd[RNMMMM��������������������CIOS[hrthh[OCCCCCCCC|~�������������||||;:9<HIUamnsnhaUTH<;;"/831/-"!)45>:5)<9<8BN[g�����tg[NHB<vyz~�����zvvvvvvvvvv}�����������������������).+)����%#&5N[g��������t[B2%YY[^ght����tg[YYYYYY��������������������fc_``hltv~���zthffffhir���������������th�������

�����IHIU`bfeb[UTUIIIIIII�������������T[gkqt����xtg[TTTTTT��������������������������
)' ���������IBBEFJN[gjtw|wth[NI�������  ��������������
#%$&#
	�VZamz��������zmjcaVV�������������������������
!0;<&%�����}yy{���������������}��������

��������� )BEFD;:3������	!()6CBEB)��")6O[hmkpnhf[OB2#opst~��������toooooo9<IUZWUI=<9999999999����������������������������������������������������������5HNONIB5��������������������������

������&+/<HMHE</&&&&&&&&&&"+/;?A<;/("[Uaz�������������{n[��������������������
).) 457=BLN[dghgg`[NB7540.026?BMO[^`]^[UPB60VWY[ghirnih[VVVVVVVV�����������������������������������upnuz��������zuuuuuu����)BHOB6)������������

���ڽ����������������������_�l�x���������x�l�a�_�W�_�_�_�_�_�_�_�_�(�2�5�A�N�R�Y�\�Y�N�5�-�(�������(ù��ùôü����ùìàÓÇ�~�~ÇÓÖÓìù�����������������������������������@�L�O�Y�^�^�Y�R�L�J�@�3�/�3�5�:�@�@�@�@����'�-�/�1�3�-�'���������������"�-�/�1�7�6�/�)�"���	��	������;�H�T�a�m�n�m�g�a�[�T�N�H�?�;�3�;�;�;�;�x���������������������������y�x�l�v�x�x�
���#�(�(�#��
�
�� �
�
�
�
�
�
�
�
�#�/�<�H�H�K�I�H�<�<�/�#����!�#�#�#�#��������
������������������������H�T�a�b�f�h�a�T�H�=�;�:�;�C�H�H�H�H�H�H�f�f�j�j�f�e�Z�M�A�A�K�M�Z�^�f�f�f�f�f�f�������������������������w�[�H�B�A�s������"�%�/�4�/�"��������������������������������������{�}�������������������������������������������������)�6�O�g�o�r�n�g�O�6������������������������������������������������������������������������������������������������һF�S�_�l�x�}�x�o�l�_�Z�S�H�F�F�B�F�F�F�F�(�4�M�\�s������y�f�b�M�A�4�(�����(���(�4�A�O�W�]�Z�V�M�A�4�(������������������������������������������������f�r��������ͼμʼ��������Y�@�5�-�4�@�f�;�<�A�G�T�[�Z�T�G�?�;�:�7�6�;�;�;�;�;�;�y�������������������������������y�y�y�y���������������u�G�.�"�������.�G�`�m����)�5�B�N�[�a�d�^�[�R�N�B�5�)�%������������������������u�s����������������M�Z�f�p�g�Z�X�H�A�4�(�#����(�2�4�A�M�"�.�1�1�.�"���� �"�"�"�"�"�"�"�"�"�"ĚĦĪįįĨĦěĚčČā�~�z�~āčĐĚĚ�����������������������������������������H�T�a�m����s�a�/����������	���/�;�H���Ľн���������ݽ�����������������DoD�D�D�D�D�D�D�D�D�D�D�D�D{DdDVDSDVDbDo�Y�e�~�������ǺȺϺԺɺ������~�r�]�S�P�Y��!�-�F�S�F�B�-�!�������ٺ�������ʾ׾�����������׾ʾ��������������ʿ	��"�$�.�0�.�%�"���	�� �	�	�	�	�	�	������������������������������������ÇÓ×ÓÑÓÕÓÍÇ�z�v�zÀÇÇÇÇÇÇàù������������ùìÓ�z�t�d�`�_�a�zÇàƳ����������������������ƳƎƇƐƔƛƨƳ²¿����������������¿¦²����������������ݿֿѿпѿֿݿ�꺗�����������ɺ����������~�v�w�~��������²¿����¿¿¼²®«²²²²²²²²²²��#�0�7�<�@�G�<�0�#�#����������ܹ����������Ϲ������������Ϲֹܽy���������������������y�l�`�U�\�`�b�o�y�#�%�0�7�;�:�0�)�#������#�#�#�#�#�#�I�K�U�b�g�n�o�n�b�W�U�I�H�<�6�3�9�<�@�I���ûлܻ������������ܻлû���������'�4�9�@�A�@�4�'����������������������������w��������������������/�;�;�;�/�/�5�5�/�-�"�������"�)�/�x�������������������x�u�l�q�x�x�x�x�x�x�������üʼȼļü��������������}�}������EiEuE�E�E�E�E�E�E�E�E�E�E�E�E~EqEoEiEhEi ; [ 6 W O 1  T > 0 ` A d . J B @ N '  S L N @ [ B T � = =   D t Y 9 / R A ' K ^ 4 3 [ V A 1 F 1 K \ M X ! W  V O < | , m B      G  �  �  V  �  �  >  �  9  �      �  �    0  X  �  ;  �  �  �  ^    Y  H  �       �  �  R    H  O  �    �  6  �  d  �  E  Z    L  J  +    n  �  1  +  �  6  �  h  t    �    ��`B�D��<D��=+<o<��
=o<u<�o=t�<�C�<�<��
<�t�<�1=\)<��
=49X=+>�C�<ě�<���<���=��=��<�9X=}�<�=o=��=Y�=�w=�P=�w=@�=H�9=�+=���>@�=�9X=y�#=}�=L��=P�`=T��=���=�;d=��`=}�=�C�=�o=u>&�y=���=�7L=���=���=��P=�-=���=�v�>/�>�RB&r�B&y�Bc�B!�GB	�Bl�B 6�B	A�B{oB"u8B*�B��B��A�u�B�#B	RpB �B��By�B	L�B	z�B�\B�Bs%B#jeB'R�B#b(B	^(B �B��BܤB�B�B(3A�7{B��B��BŰB �B�B��B��B�$B&��B��B�B��B�.B�B#��B��A�^�B��B,
B��B@�B�#B�{B�hB��B��B��B0�B&DJB&~Bv
B"?B	BBC�B�vB	�)B�B"��BI�B��B~�A�SB��B	��B ?�B�4B?�B	?DB	K�B){BœBvgB#�XB'��B#AHB	�`B
��B�B�`B�`B@�B=.A���B��B�B�B<5B��B�IB�+B�0B&��B\�B=�B��B��B��B#��BO�A�}lB�B,ZB�BE{B@�BW�B�;BV�BM B=�B~�A/t@�� A���AːUA��?�W�?u@�A�(�A�U�@�rA�.bA¾>A�'QA��ZA>��A��HA�RA��TA���A�K�A��A�(}@�tgA;g�A7�,@��@�c�Ae-kAq��Aaz�A��0AG�cA;EA_�A�A�A�יA��7A'S�C��%@�o@b
�AR`5A]-@QK�AəoA�wdB�@A���A2�@-A��A��>�P:A��A���A�O@�?}@ʺ,@ꗲA��E@��@�C�[A.��@��2A��AA�t2A���?� �?n��A���A�N�@�*A��DAAҋ~A���A?�A�{�A��A��A�(�A�A�v�AЁ�@�A9��A8�@�[@���Ae!Ap��AaYA��AH"WA:�KA^�CA߄A���A��3A&�|C��8@4�@l�-AQ	A]��@L`�A�}�A�VB>NA��AD@K A��A�S>��>A�4A�oA�|s@���@ɜ�@�
xA��@��@��C�p                                        
                    }      
   
            )         9                        +   �   /                  4   <   5         
      k            $               O   4                                                %      '      1            '         %         +                     1   %   !   #   '               #   #   %               -                           %                                                         %      !            %                                       +         #   '                                                                  M��wNfO&=OrrN<�N�٦OWYN�&N�>�N�9N��ENJ�N�ȰN[wN���O���N9�O�EZO0�rO�N��	N|��N��~O��Oec�N<
mNq�ZNP�mN��dO�OE}9N�s1N��M���O;"O�P|O!TOJ�QOŶbO�x�O��N�"�M�$�N$�O��vO��O��N�T@N���N��N�zlO�JWN��yN���O'O3�1NE�{NX�N�",N�X�O}!O]iB  2  #  �  �  �    z  :    q  �  �  �  R  �  �  -  a    �  �    �  �  �  �  �  �    q    a  X  -  �  >      w  �  <  c  �    �  	K  ,  �  R  #  V  z  E  �      �    �  �    A  ��+���
:�o;ě�;�o<o<o<t�<t�<�C�<#�
<�o<49X<u<u<�t�<�C�<���<�1>L��<�C�<�C�<�C�<��
<���<��
=H�9<���<�/=m�h<��=o=o=\)=t�=t�=�P=aG�=��=@�=,1=49X=49X=8Q�=D��=y�#=y�#=ix�=T��=P�`=aG�=e`B=� �=�t�=y�#=y�#=�o=�o=��P=���=��=���=�
=:5:<ILNI@<::::::::::;:9<@IJKKIF<;;;;;;;;z|����������������zz��������������������dbfgitx~{tmgddddddddcehot��������xthcccc��������������������JGNS[gt|����|tqg[NJJMGIJNNP[agigd[RNMMMM��������������������CIOS[hrthh[OCCCCCCCC��������������������;:9<HIUamnsnhaUTH<;;"/51/&")15<85)A>BN[g������tg[NLJFAvyz~�����zvvvvvvvvvv��������������������������(&#���@?BGN[gt������tg[KD@YY[^ght����tg[YYYYYY��������������������fc_``hltv~���zthffffjks���������������tj�������

�����IHIU`bfeb[UTUIIIIIII������������������T[gkqt����xtg[TTTTTT������������������������������	

����JCCFGLN[gitv{~wtg[NJ�������  ��������������
#%$&#
	�VZamz��������zmjcaVV�������������������������
 /::/�����������������������������������

���������6ACB=98/������	!()6CBEB)��")6O[hmkpnhf[OB2#opst~��������toooooo9<IUZWUI=<9999999999��������������������������������������������������������� )5EKLJB5)���������������������������

������&+/<HMHE</&&&&&&&&&&"+/;?A<;/("����������������������������������������
).) 457=BLN[dghgg`[NB7540.026?BMO[^`]^[UPB60VWY[ghirnih[VVVVVVVV�����������������������������������upnuz��������zuuuuuu�������')$������������

��۽����������������������_�l�x���������x�l�a�_�W�_�_�_�_�_�_�_�_�*�5�A�N�P�X�[�Z�X�N�5�/�(������&�*àæìøù����ùìàÓÇÂÁÇÒÓÚÝà�����������������������������������@�L�O�Y�^�^�Y�R�L�J�@�3�/�3�5�:�@�@�@�@����'�-�/�1�3�-�'���������������"�-�/�1�7�6�/�)�"���	��	������;�H�T�a�m�n�m�g�a�[�T�N�H�?�;�3�;�;�;�;�����������������������������y�����������
���#�(�(�#��
�
�� �
�
�
�
�
�
�
�
�#�/�<�D�A�<�2�/�)�#�!�"�#�#�#�#�#�#�#�#��������
������������������������H�T�a�b�d�f�a�T�M�H�D�F�H�H�H�H�H�H�H�H�M�Z�f�i�i�f�b�Z�M�F�D�M�M�M�M�M�M�M�M�M���������������������~�s�`�N�I�T�g�s������"�%�/�4�/�"���������������������������������������}������������������������������������������������������)�6�A�K�O�M�D�6�)���������������������������������������������������������������������������������������������һF�S�_�l�x�}�x�o�l�_�Z�S�H�F�F�B�F�F�F�F�(�4�M�Y�f�s������v�f�_�M�A�4�(����(���(�4�A�O�W�]�Z�V�M�A�4�(���������������������������������������������������������������r�r�n�r����������������;�<�A�G�T�[�Z�T�G�?�;�:�7�6�;�;�;�;�;�;���������������������������{��������������"�.�;�G�M�T�X�T�Q�G�;�7�.�"�������)�5�B�N�[�^�b�]�[�Q�N�B�5�)�&������������������������u�s����������������M�Z�f�p�g�Z�X�H�A�4�(�#����(�2�4�A�M�"�.�1�1�.�"���� �"�"�"�"�"�"�"�"�"�"ĚĦĪįįĨĦěĚčČā�~�z�~āčĐĚĚ�����������������������������������������T�a�m�����~�r�a�/����������/�;�H�T�����Ľнݽ�����ݽнĽ�������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DwDjDjDoDtD{�Y�e�~�������ºúͺɺ��������~�r�`�W�S�Y��!�-�F�S�F�B�-�!�������ٺ�������ʾ׾�����������׾ʾ��������������ʿ	��"�$�.�0�.�%�"���	�� �	�	�	�	�	�	������������������������������������ÇÓ×ÓÑÓÕÓÍÇ�z�v�zÀÇÇÇÇÇÇÇÓàù����������ùìàÓÀ�z�m�i�l�zÇƳ��������������������ƳƧƛƗƗƗƛƤƳ¿��������������������¿¦²¿�ݿ������������ݿؿѿ׿ݿݿݿݿݿݺ������������ɺ����������~�v�w�~��������²¿����¿¿¼²®«²²²²²²²²²²��#�0�7�<�@�G�<�0�#�#��������������	��� ����ܹϹù��������ùϹ۹�y�����������������|�y�o�r�w�y�y�y�y�y�y�#�%�0�7�;�:�0�)�#������#�#�#�#�#�#�I�K�U�b�g�n�o�n�b�W�U�I�H�<�6�3�9�<�@�I���ûлܻ������������ܻлû���������'�4�9�@�A�@�4�'����������������������������w��������������������/�;�;�;�/�/�5�5�/�-�"�������"�)�/�x�������������������x�u�l�q�x�x�x�x�x�x�������������żǼƼ¼�������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E�E~EqEoEiEu ; [ 4 G O 1  T > , ` : d < G < @ J *  S L N = [ B ' � B (  D t Y 9 / O . " N ^ 4 3 [ V > ! / ( K \ M # J W  V O < | , B <      G  f  9  V  �  �  >  �  �  �  [    w  �  �  0    v  �  �  �  �  1    Y  s  �  �  K  �  �  R    H  O  �  ^  �  �  �  d  �  E  Z  X  �  �  �    n  �  j  �  �  6  �  h  t    �  	  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  2  ,  %      
  �  �  �  �  �  �  |  ]  >     �   �   �   �  #  #  "  "  !  !             !  &  *  /  4  8  =  B  F  K  �  �  �  �  �  v  h  U  ?  $  	  �  �  �  �  g  ?    �  a    �  �  �  �  �  �  �  S  :  :  3    �  �  ;  �  |    �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  e  T  B  1             �  �  �  �  �  �  �  �  �  �  k  <  �  �  G  �  !  z  v  m  b  U  :    �  �  �  �  �  �  �  s  A    �    �  :  7  4  1  #      �  �  �  �  �  �  �  �  �  �  �  j  9            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     *  N  f  p  m  b  P  @  "  �  �  �  �  `  `  �  �  �  �  �  �  �  �  �  �  �  x  n  f  n  v  W    �  q  7   �  '  [  r  �  �  �  �  �  �  l  R  6    �  �  A  �  �  +  �  �  �  �  �  �  �  �  �  �  �  �  a  .  �  �  �  k  =    �  :  ?  D  I  N  P  L  H  D  ?  7  *        �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  d  N  ,    �  �  �  h  *   �  �  �  �  �  �  �  �  �  p  V  7  
  �  �  E  �  �  �  ^  +  -  $      
    �  �  �  �  �  l  O  0     �   �   �   �   r  B  U  `  V  Q  Z  \  ]  Q  A  !  �  �  �  U    �  �  !  �  �  �              �  �  �  �  �  h  2  �  �  q  5  	  _      �  �  �  �  �  T  �  u  �  -    �  �  �  E  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w          �  �  �  �  �  �  j  K  +    �  �  �  �    �  �  �  �  �  �  x  g  T  B  /    
  �  �  �  �  e     �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  @  �  r   �  �  �  �  t  l  c  X  ;    �  �  �  u  A    �  �  J       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  D  Q  n  {  �  �  �  �  �  m  i  �  �  �  �  �  O  �  �  �  �  �  �  �  �  �  �  �  x  e  R  ?  )    �  �  �  �  �  w  �  �  �      �  �  �  �  �  �  �  �  y  d  O  7     �   �  �  �  "  :  A  9  )    �  �  m  p  ]  6  �  �    �  �  �          �  �  �  �  �  �  d  C    �  �  ~  J  �    ;  a  [  U  N  E  <  0  !    �  �  �  �  �  k  H  $  �  �  �  X  Q  J  C  >  B  E  I  I  E  B  >  0      �  �  �  w  M  -  %          �  �  �  �  �  �  �  �  �  n  Z  F  1    �  �  �  s  \  C    �  �  �  a  3    �  �  �  �  ]  /     >  1  "    �  �  �  �  �  y  V  0  
  �  �  �  �  �  m  X  �    �  �  �  �  �  �  �  �  �  f  8      �  �    �   �  |  }  �  �  �  �  �  �       �  �  �  �  ~  -  �     w  �  e  r    �  �  <  f  v  h  8  �  S  �  �  x    �  5  
�  2  �  �  �  �  �  �  g  ;    �  �  ,  �  �  �  �  0  �  \  �  <  +      �  �  �  �  �  �  `  >    �  �  �  O    �  ]  c  ^  b  T  A  *    �  �  �  �  Z  -    �  �  �  Y  2    �  �  �  �  �  �  �  n  W  A  %    �  �  �  z  V  2     �      �  �  �  �  �  �  �  �  �  v  [  8    �  �  o  ;    �  �  �  �  �  �  �  �  �  �  �  �  �  �    v  m  e  \  S  �  �  	  	?  	J  	I  	3  	  �  ~  '  �  >  �  �    4  �  �  J  �  �    %  +    �  �  �  q  <    �  �  X  �  u  �  �  �  n  �  �  �  �  �  o  [  I  6    �  �  b  �  i  �  &  _  �  .  C  O  C  7  +        �  �  �  �  w  D  �  �  6   �   ~  #      �  �  �  �  o  C    �  �  H     �  ~  C    �  �  V  e  u  y  }  �  �  �  �  �  �  �  �  �  �  �  �  �  o  R  z  i  X  G  6  %      �  �  �  �  �  �  �  �  �  �  �  �  
�  
�  �    0  D  B  7    �  �  p    
{  	�  �  �  �  �  �  �  �  �  �  $  Y  �  �  �  �  �  �  �  �  �  �  i  K    &    �  �  �  �  �  �  �  �  z  g  S  >  /  +  '  #           y  k  X  B  +    �  �  �  �  }  Z  4    �  �  F  �  �  �  �  �  �  �  }  �  �  �  �  y  V    �  -  �    w  �         �  �  �  �  �  d  F  '    �  �  �  �  \  ;    �  (  �  �  �  �  �  n  \  J  4      �  �  h  (  �  �  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  v  Y  =       �  �  �  �  \  6    �  �  �    =  -    �  �  �  >  �  I  
�  	�  	O  �  �  ;  B  �  S  �  �  o  W  C    �  2  
�  
<  	�  	3  �    b  �  �  �  '