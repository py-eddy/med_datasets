CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�bM��     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Q%   max       P��     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��{   max       =8Q�     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @FU\(�     �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @v|Q��     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q            �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @���         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �1'   max       =o     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B06�     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��>   max       B02�     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�3�   max       C���     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�tC   max       C���     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          _     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Q%   max       PtPv     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��2�W��   max       ?� [�6�     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       =8Q�     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @FP��
=q     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @v|Q��     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�'          (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?l   max         ?l     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?��_o�        `�                     &                   :                     ,   ^   6                        
                               #             &                                    S         (   	       %   	         O      ?                   O%��O�n�N�.$N��N@&_O+O+�N�1N�,Nt��N1��P,j\Pv�EN<�&OT��Oz��O�aN�ĊO�K�P�P��P��RNG�M�J�O���Nc�OOD��ONd�N{BOO�N�T4N�:oNʶ^O��2O��6OGw�O!�9O���N�m�P
��N?2�O��\M�Q%OnS�N�q.N}�O9�'O�`�N�]O0t�O��#N��*Oz��N��5NrsuP��O�C�N�p�Oy�PN:�O�Q]OQg�N��N9�(N���O�D�OB�O��N'��Oj�ON^O��O��O2�=8Q�<�t�;�`B�o�o�D����o���
���
�o�o�t��t��t��#�
�49X�49X�D���D���e`B��o��C����㼣�
��1��1��9X��9X��j��j�ě����ͼ���������/��/��`B��h��h�����������o�+�+�+�+�\)�t�������w�'',1�49X�8Q�8Q�@��L�ͽT���Y��aG��aG��e`B�e`B�e`B�������罩�置{��{��������������������_hmz���������ztjia]_))6;BOOSVVOKB;6-)'")��������������	������������������������������
#/9=?<3#
�����GNS[gt����tg[RNGGGG
 
#$#/2<=?<9/###%/<D></# �����������������������
#<DFC3#
������������������������������������������������5@FJMONB5)
45>BGN[^\\[XSNJB65+4�����
����������
�����������-/,�����x�����������������yx#Ib{������{U<0
����������������������

������������y~��������������zvvy������������������������������������������
#<AHHA</ ��&))688630)ot��������������tpjo:;@CFHIOTXZ\\WTLH;::������	�������������),58BHLJB;50))*)'&&)���*6COmsph\C���45=BNS[gt����tg[NB84���������������������������������������pt��������������{qmp����������������������������������""
��������������������`anozzzna_``````````#%0<IIPUXZXUK<0-)$"#%)46BFCB6.)����������{{����������������������������)6BO[sxwtnh[OC9)~���������������x{~~��������������������Ubnx���������{ndb[VU�������� ����������������#0#
�������stz��������topssssss4;HLNLH;004444444444#/<OOUnz��nH</##��������������������.01;<DGIRSSPLIA<0/..nz�����������zqngehn��������������������~�����������������}~)5=BIMSVNB5)##*,/+#��� �����������������������������������&-22,������������������������������-/;99/'!
���X[dhhopkhb[[XXXXXXXX��������������������)05<BNYYWUTPNB?5.)()gt���������tg`bdc`_g���������������������������������;�/�#���
��������
��#�/�:�<�D�H�=�;ÓÇ�{ÀÊÔì�����������������ìàÓ�:�9�7�:�=�F�F�S�l�x�{�x�s�l�h�_�W�S�F�:�t�o�t¦²¶²§¦�B�@�5�,�5�?�B�E�N�[�\�[�V�N�B�B�B�B�B�B�4�(�"���(�.�5�A�N�Q�Z�`�Z�X�N�J�A�5�4����������$�0�=�I�T�R�I�@�2�0�%���������������	�
����	����������������������������������������������������u�s�g�e�g�p�s���������������������������������������������������������������˾ʾ��������ʾ���"�.�@�O�T�;��	������	��������5�[�g�g�N��	�"� ������"�"�+�*�/�0�/�"�"�"�"�"�"�y�r�m�m�e�l�y�������������������������y�m�f�b�o�y���������������������������y�m������x�s�m�n�s������������������������h�d�[�O�B�4�6�B�O�[�d�h�t�~āĄā�t�h�h���������������������������������������׾����׾¾����ʾ��	��"�B�M�N�F�;�.����ɺ����_�[�h�����ֺ�-�F�_�S�;�*�!��ֺ������q�I�7�&�1�S������������������������������������������������������������������ü������������������������������������ā�t�e�F�B�;�B�K�[�h�kāčĦĠĢĝĚčāŹŷŭŠŜŠŭŹ����������������ŹŹŹŹ��������������������������������������	�����&�)�6�B�O�Y�T�O�J�B�6�)���s�l�f�Z�Y�P�Z�f�l�s�������s�s�s�s�s�s��������������������������������������������������v�s�g�^�g�s�����������������������������(�4�4�A�A�A�4�(������%�(�1�5�A�N�W�Z�^�Z�O�N�A�<�5�(���G�@�;��	� �����"�.�;�J�O�N�G�I�L�I�G�*����������#�6�C�O�[�_�]�U�I�6�*�ѿǿĿ����������Ŀѿݿ�������������n�h�c�n�uŇőŔŠūŭŴůŭŪŠŔŇ�{�n�ѿͿʿϿԿԿڿݿ������������ݿѺ����������ɺӺֺ׺���ݺֺɺ��������������������������������(�/�/�$�����ŠśŚŠŭŹž��ŹŭŠŠŠŠŠŠŠŠŠŠ�b�Y�[�_�\�V�L�V�b�o�{ǈǔǢǧǥǡǔǈ�b�z�w�y�zÆÇÌÌÍÇ�z�z�z�z�z�z�z�z�z�z������������(�4�A�M�W�S�M�A�(�����ܹ۹ܹܹ߹������������������	��	���"�.�/�/�/�"��	�	�	�	�	�	�	�	ā�t�h�c�f�h�tāĚĦĳĿ����ĿļĳĦĚā���r�r�u�����������������������������������x�s�g�Z�U�V�Z�g�s���������������������H�E�<�8�<�=�H�U�a�n�o�v�z�}�|�n�k�a�U�H�����������@�M�W�Y�`�c�h�Y�M�@�'�ƚƎƍƅƌƎƚƛƧƳƵƶƳƭƧƚƚƚƚƚ�������������������������ʽѽҽ׽нĽ�����������������������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�FFFF!F,F FE�E�E��
�������������
��#�0�<�I�U�^�U�I�#�
�l�c�_�S�Q�F�:�8�:�F�S�_�l�q�x��z�x�l�l�ù��������������ùܹ�����������ܹϹ�FFFFFF$F0F1F9F1F$FFFFFFFFF�3�'�#��'�-�5�=�L�e�r�~�������r�Y�L�@�3��������������������� � �������뻷�������ûлܻ��ܻлû����������������Ľý��Ľнؽݽ��ݽؽнĽĽĽĽĽĽĽĻû������������ûлڻѻӻлǻûûûûûûܻܻ������4�@�M�Y�b�f�b�Y�M�4��������(�5�8�A�N�S�N�L�7�5�(������ED�D�D�D�D�D�D�D�D�D�EE*E7ECEFE=E.EE�T�I�H�G�H�T�a�m�w�m�c�a�T�T�T�T�T�T�T�T�����������������������������������������n�a�Z�U�Q�P�U�a�n�zÇÓ×ÖÓÍÇ�}�z�n������������#�/�H�]�_�X�S�H�<�#�
��������ڼ�������������������!�������!�.�C�G�I�L�J�G�B�:�.�&�! S T R S F ) ) P F % ] U & � 6 = C Y L J @ = f e = z 8 k n J y M ` z 3 O @ L P C B J _ 1 = ) \ Q \ ) * E G G & I F h J J 3 ! 0 A H B ; X H 5 . \ / .  �     5  �  S  E  �  <  <  u  {  #  �  �  �    ;  6  �  k  �  a  Y  ?  �  �  �    �  f  &  
    �  t  �  k  �  �  �  Q  P  $  �  �  �  �  �  �  w  w  �  >  �  r  �  �  �  '  f  �  �  �  Y  �  /  \  �  J  X  Y  �  G  *=o�D���T���D���ě��T���'T���o�T���e`B�'�+�49X��j�C���t������P�ixս�G���hs��1�ě��L�ͼ�j�+�49X��/�+��`B�t����]/�D�����49X�y�#�\)��+��P��%�+��O߽8Q�#�
�H�9��%�8Q콅��H�9�aG��q���8Q�e`B���m�m�h�@���{�]/���w��-�u�ixս}�1'��-����q�������ȴ9��xս��ͽ�^5B�]A��HB�B��B@�B�yB�QB	L$B��B9�B3�B�<B�B �B3BƐB}B-�B��B/�B��B'=B�Bx{B'�BTWB��B�~B=MBK�A�}�B�_B>�B06�B�B*�Be�B=B!v1B�TB�AB�lB�3B&F-BjB g=B�WB@�BZ�B!/B)�B�1B#K.B
XiA��B	�B��B&m�B��Ba�B ��B:B%
�BE�B��BԜB�B*hB��B>�B��B	��BC�B�B�	B X
B3�B��B?�BUB�<B	E�B��BB�B:iB@B9B �B��B�8B MB�B��B1�B� B'��B��BO�B?�B=FBu`B �BDB�FA��5B=@BoeB02�B��B)��B��B
��B!B\B�/B��BBkB~B&@!B�oB >(B�tB6�BJyB!�B)�B�B#t�B
@A��>B��B��B&i�B�"BX7B ��B�dB%'�B<�BQgB�HB�5B�B��B}QB�bB
B�B@!B��A�SA�@n@��kA� �A���A�(�B	�#AY�A�G�A�)BA�nbAW�hA��A�s4AqqfAo�#AG�GAڬA���AZ�<@=(�A���A��A�l�A�֐A��/B9�A�4�AB5G@�A�A�
MA5�$A��3Aa�?B AA{�A� A@4kA�+�A���Bq�A�BvA6a�?"abA��Aް
A��A���A�
G@�FBm�A$�A��BM�C�v�A�Q@���>�3�C���?�+rA�f@���A(�@�~@���A�o2C�U�A�#�A�#A���A�7�A:�A�A�A͎�@��vA��A���A�l�B	�AX��A�t1A��2A��1AURA���A��sAq��Ao8AH[�Aڂ�A�{AZ�@K�jA�b�A��AΒJA�O�A�`�Bo�A�u�AD�@�A���A4	�A���A_��B /QA}w�A�DA�rr@-׎A�EA��B��A�c5A7�?+��A���AݑA���A�=�AŁ@��B�;A#�A�v�B@C�h�A�>@�g�>�tCC���?��A��@�ȱA)@�u@���A�)�C�O�A�_\A�m�A�k|A��iA�	A=�                     &   	            !   :                     ,   _   7                                                !      $      !      '                                    S         )   
       &   	         O       ?                         %                              -   /                     '   A   9                                    '   !               %                        #                        -   !            #               %      #                        %                              )   '                        5   1                                                                                                                                                            O%��O��;N�.$Np��N@&_O ��O �N��N�,Nt��N1��P��P49N<�&N���N��O�aN�ĊOX�rO�1PtPvPj�NG�M�J�Oli�Nc�OOD��O�WN{BOO�N�T4N�:oNʶ^O��O��EOGw�O¢O�
�N�m�OQ�N?2�O1M�Q%OZ,�Nt�N}�OE�O{�N��(O5�Ow.dN��*OW��N��5NrsuOn�QO���N�p�ON 4N:�O�r�O�nN��N9�(N���O�,lO�Ol�N'��Oj�ON^O��O>�O2�  
  �  =  \  �  	  �  �  J  	  �  �  {  �    ?  W    Z  7  Z  �  [    �  �  z  D  b  �  G    S  �  3    �  �  :  �  !  :  h  z  �  �  L  X  V  �  o  �  d  �  �  �  Y  �    �  �  �  �  _  �  
�  $  �  R  �  f  �  �  |=8Q�<�C�;�`B��o�o��o��t���`B���
�o�o�e`B�����t��e`B���
�49X�D����C���`B��󶼴9X���㼣�
������1��9X���ͼ�j��j�ě����ͼ�����h��h��/��h����h�@�����P���+�C��+�\)�#�
�C�����w���#�
��w�'���8Q�49X�L�ͽ8Q�e`B�m�h�T���Y��aG���+�ixս�C��e`B�������罩�罰 Ž�{��������������������`imz��������ztrmib_`))6;BOOSVVOKB;6-)'")����� �������������	����������������������������#/1684/%#P[\gt{���}tg[WPPPPPP
 
#$#/2<=?<9/###%/<D></# �������������������������/9=2)#
����������������������������������������������)57>BBB95)45>BGN[^\\[XSNJB65+4�����
���������
����������� ��������������������������#Ibn{�������{<0����������������������

������������������������������������������������������������������������ 
#/<=EE><*#
 &))688630)ot��������������tpjo:;@CFHIOTXZ\\WTLH;::������	�������������),58BHLJB;50))*)'&&)�*6COhole\OC6��9BN[gt����~tg[NB?979���������������������������������������qt�������������|trnq����������������������������������������""
��������������������`anozzzna_``````````#$'0<INUXYWTI<0.*(%#))+6BA6)����������{{����������������������������)6BO[bhkihb[OK?)��������������}��������������������^bn{~��������{nh``]^�������� �����������������
��������stz��������topssssss4;HLNLH;004444444444#/<?@EHC</)#��������������������.01;<DGIRSSPLIA<0/..jpz�����������zunkhj����������������������������������������$)5BBINOQNB75)#*,/+#��� ����������������������������������� (.-&����������������������������
#/2/,(# 
������X[dhhopkhb[[XXXXXXXX��������������������)05<BNYYWUTPNB?5.)()gt���������tg`bdc`_g���������������������������������;�/�#���
��������
��#�/�:�<�D�H�=�;ÓÇ�~ÁËÕì�����������������ìàÓ�:�9�7�:�=�F�F�S�l�x�{�x�s�l�h�_�W�S�F�:�y¦®¦£�B�@�5�,�5�?�B�E�N�[�\�[�V�N�B�B�B�B�B�B�(�%�!�"�(�4�5�A�N�O�Z�]�Z�W�N�H�A�5�(�(��	�����$�0�=�B�I�K�I�E�=�0�$����������������	����	�������������������������������������������������������u�s�g�e�g�p�s���������������������������������������������������������������˾������������ʾ��/�G�M�;�"�	�����׾��������&�5�[�t��t�g�[�5��"� ������"�"�+�*�/�0�/�"�"�"�"�"�"�����y�t�y�y�����������������������������y�t�q�x�y�����������������������y�y�y�y������x�s�m�n�s������������������������h�d�[�O�B�4�6�B�O�[�d�h�t�~āĄā�t�h�h���������������������������������������˾���Ծ̾׾�����"�.�9�=�;�2�.�"��	���������o�g�u�������-�@�9�3�/����ɺ����u�O�>�9�9�D�]������������������������������������������������������������������ü�������������������������������������t�n�[�K�B�>�B�O�[�h�s�tāċčĘĔčā�tŹŷŭŠŜŠŭŹ����������������ŹŹŹŹ������������������������������������������#�(�)�6�B�O�S�O�N�F�B�6�0�)��s�l�f�Z�Y�P�Z�f�l�s�������s�s�s�s�s�s��������������������������������������������������v�s�g�^�g�s�����������������������������(�4�4�A�A�A�4�(������%�(�1�5�A�N�W�Z�^�Z�O�N�A�<�5�(���.��	������"�.�;�G�L�M�K�I�D�E�D�;�.����
���&�*�6�C�O�X�]�[�R�G�C�6�*��ѿǿĿ����������Ŀѿݿ�������������{�o�n�e�n�vŇœŔŠũŭŴŮŭŨŠŔŇ�{�ѿο˿пҿտֿݿ�������������ݿѺ����������ɺӺֺ׺���ݺֺɺ�������������������������������� � ������ŠśŚŠŭŹž��ŹŭŠŠŠŠŠŠŠŠŠŠ�o�h�c�b�f�h�o�{ǀǈǔǟǡǣǡǟǔǈ�{�o�z�w�y�zÆÇÌÌÍÇ�z�z�z�z�z�z�z�z�z�z������������(�4�A�M�U�Q�M�A�4�(����ܹܹܹ޹�������������������	��	���"�.�/�/�/�"��	�	�	�	�	�	�	�	�h�f�h�tāčďĚĦįĳĶĶĳĦĚčā�t�h�����x�v�z���������������������������������{�s�g�Z�V�W�Z�g�s���������������������H�>�<�:�<�@�H�U�a�m�n�t�|�z�x�n�i�a�U�H����������@�M�V�[�\�X�M�@�4�'�ƚƎƍƅƌƎƚƛƧƳƵƶƳƭƧƚƚƚƚƚ���������������������ĽǽϽннսнĽ�����������������������������������������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�FFFFFE�E�E�E���
��������
��#�0�<�C�I�R�T�I�<�0�#��l�c�_�S�Q�F�:�8�:�F�S�_�l�q�x��z�x�l�l�Ϲù������������ùܹ������������ܹ�FFFFFF$F0F1F9F1F$FFFFFFFFF�3�/�,�1�7�:�>�K�Y�e�r�~�������~�r�L�@�3��������������������������������������ûлܻ��ܻлû����������������Ľý��Ľнؽݽ��ݽؽнĽĽĽĽĽĽĽĻû������������ûлڻѻӻлǻûûûûûü���������4�@�M�V�^�b�^�Y�M�@�4����������(�5�7�A�N�Q�N�K�6�5�(��D�D�D�D�D�D�D�D�EEEE*E8E7E*E'EED�D��T�I�H�G�H�T�a�m�w�m�c�a�T�T�T�T�T�T�T�T�����������������������������������������n�a�Z�U�Q�P�U�a�n�zÇÓ×ÖÓÍÇ�}�z�n������������#�/�H�]�_�X�S�H�<�#�
��������ۼ�������������������!�������!�.�C�G�I�L�J�G�B�:�.�&�! S P R < F , * H F % ] [ ' � = $ C Y A F = 5 f e C z 8 e n J y M ` d - O = F P 8 B 2 _ 0 < ) U T Y * ' E 0 G & R M h @ J 9  0 A H * : G H 5 . \ ' .  �    5  r  S     (  �  <  u  {  �  �  �    �  ;  6  �  I  !  �  Y  ?  �  �  �  o  �  f  &  
    �  .  �  Q  f  �  �  Q  �  $  �  �  �  P  ?  �  Q  �  �  �  �  r    /  �  �  f  Q  <  �  Y  �  �  R  �  J  X  Y  �    *  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  ?l  
    �  �  �  �  �  �  �  �  �  �  �  �  �  }  k  G     �  �  �  �  �  �  �  �  �  �  m  N  %  �  �  _    �  v  	  �  =  0       �  �  �  �  �  �  �  �  �  �  Z  %  �  �     q  /  -  /  H  X  G  4      �  �  �  �  Y  %  �  �  �  V  $  �  �  �  �  �  �  �  �  �  }  j  P  5       �  �  �  �  �            �  �  �  �  �  �  �  p  W  E  4    �  �  �  %  d  �  �  �  �  �  �  �  �  z  N    �  �    c  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  f  G  (  �  �  W  
   �  J  F  B  >  :  6  2  +  "        �  �  �  �  �  �  �  �  	  	            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  P  r  �  �  �  �  �  �  �  �  �  �  �  �  i  (  �  7  �  O   �  �  �    8  ^  u  {  c  7  �  �  4  �  D  �  �  G  �  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  v  q  �  �  �              �  �  �  �  �  �  `  =  %      �  �  �    "  -  4  <  >  6  #    �  �  b    �    y   �  W  R  M  G  A  :  6  4  3  -  '  !        �  �  �  �  �    	    �  �  �  �  �  �  �  �  �  �  �  �  �        �    ;  R  Z  Z  V  J  7  !    �  �  �  v  H    �  �  X  <  :  �  �  �    -  7  3  $    �  �  �  g    �  `  �  t  �  �  ;  P  Z  Q  3    �  �  �  9  �  �    �    �  �    b  �  �  �  �  �  �  �  y  M     �  �  k    �    {  �  �   �  [  S  K  D  <  4  -  %              !  %  (  ,  /  2    
      !  %             �  �  �  �  |  a  F  +    f    �  �  �  �  �  �  v  T  .    �  �  T    |    �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  {  n  `  S  E  8  *  z  w  s  l  c  X  J  ;  )    �  �  �  �  a  2    �  �  ]  
       =  .      �  �  �  �  �  �  r  A  �  �    �    b  s  �  �  �  �  �  �  �  �  �  �  �  �  z  r  j  b  Z  R  �  �  �  �  �  �  �  �  �  �  }  c  F  %     �  �  �  Z  .  G  @  8  1  *  "    	  �  �  �  �  �  �  �  �  �  �  �  t            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  S  I  @  6  .  )  #      
  �  �  �  �  �  �  �  �  �  x  "  j  �  }  p  `  S  B  ,    �  �  �  k  -  �  �  9  �  �    0  2  /  '      �  �  �  �  �  b  2  �  �  �  P  �  ~        �  �  �  �  �  �  t  V  6    �  �  �  �  X  %   �  �  �  �  �  �  �  �  �  c  ;    �  �  �  c  8  
  �  �  �  �  �  �  �  �  �  �  �  d  =    �  �  j  +  �  g  �  k  b  :  5  1  ,  $        �  �  �  �  �  �  �  �  �  �  �  �  8  P  c  m  y  �  �  �  �  �  �  �  v  P  "  �  �  4  �  P  !                   �  �  �  �      *  B  ~  �  �  �  �  �  7  9  3  1  +    	  �  �  j    �  -  �    P  �  h  \  Q  F  ;  0  $        �  �  �  �  �  �  �  �  �  �  d  z  u  i  Z  B  %    �  �  �  �  q  :  �  �  �    !   r  �  �  �  �  �  x  i  Z  K  ;  *      �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  o  Z  D  ,    �  �  �  �  �  o    3  F  F  =  6  0  +      �  �  �    Y  (    �  �  �  *  >  K  S  W  W  N  9    �  �  �  ~  J    �  Q  �  e  �  D  O  S  M  C  5       �  �  �  �  �  r  _  N  ?  6  B  ]  �  �  �  �  |  Y  0     �  �  G  �  �  S  �  �    i  �  �  \  b  h  m  n  m  i  c  W  H  7  "    �  �  �  �  F   �   �  �  f  >    �  �  �  S    �  �  [    �  �  R    �  �  g  "  *  `  L  7  "    �  �  �  �  �  Z  '  �  �    H    �  �  �  �  �  �  �  �  �  �  �  �  a  C  $    �  �  �  �  �  �  s  U  8    �  �  �  �  y  V  &  �  �    ;  �  �  f    �  �  �  �  �  �  �  �  �  �  H  �  "  
K  	,  	  �  �  �  �  U  Q  Q  U  X  S  L  B  3     	  �  �  �  |  J    �  �  �  �  �  �  �  �  �  �  �  �    u  l  b  Z  T  N  I  C  =  7  �        �  �  �  �  i  -  �  �  Z    �    �  �  v   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  3  r  �  �  �  �  �  �  �  �  �  �  �  ]  *  �  �  w  %  �  :  �  1  j  ~  �  �  �  k  E    �  �  G  �  �  =  �  3  s  �  �  �  �  �  �  �  �  �  �  �  �  �  }  j  O  5    �  �  ?   �  _  Z  T  O  J  E  B  @  >  ;  0      �  �  �  o  3   �   �  �  �  �  �  |  a  C  "    �  �  �  �  �  ^  >    �  �  �  
�  
�  
�  
�  
�  
�  
�  
M  	�  	�  	G  �  m  �  g  �      �  p  !      �  �  �  �  z  L    �  �  Y    �  }  1  �  �  o  5  H  _  z  �  n  @    
�  
v  
9  
  	�  	�  �  :  [  ~  �  �  R  @  .      �  �  �  �  �  �  �  �  i  E  "   �   �   �   �  �  p  Z  =  "    �  �  �  �  f  ?    �  �  �  y  ?  �  z  f  d  `  X  L  :  )    �  �  �  �  �  _  <    �  �  �  V  �  z  r  d  M  /  	  �  �  �  z  P  B  ,  �  W  �  �    N  �  �  �  �  �  t  d  M  ,    �  �  �  Q    �  �  >  �  y  |  r  h  ^  U  L  C  A  C  D  >  /  !    �  �  �  �  �  �