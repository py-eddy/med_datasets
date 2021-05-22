CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�333333       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�N�   max       P��>       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       <49X       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F��
=p�       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v��G�{       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @3�        max       @P@           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�`           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       ;�o       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B3�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��5   max       B4�       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >ڳ7   max       C���       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�U�   max       C��<       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          L       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�N�   max       Pq�*       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�T`�d��       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <49X       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F��
=p�       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���P    max       @v�=p��
       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @M�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɖ        max       @�x�           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E'   max         E'       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��vȴ9X   max       ?�R�<64     0  ^                  *            K            	                  .            #                              C                     /                     5   2               $   #         	         5   (   
               1   #      N���M��DO��NtO�pMP*_�NXx
N҈OcZP��uO1�VO7��N'��N��JN+�!O}��OM[�N7�aO���P.��N�JOd`FN3�iO��GO�wN���N�2Oj~iN�6O4rM�N�N�irN�wP��>O��OdG<O}�O#��NB��N4H�O�K�OP��N-(N6K�P
�N��N �*Px�O��KN�e�N??iN��-O��OS��Obb*NB�O��:Nδ�O1��N�TzP>O��O/�DO�eNߤOrtN�VO;vO3�vN��@NJ�<49X��o�D����o��`B��`B��`B��`B�#�
�#�
�D���e`B�e`B�u��o��o���㼣�
���
��9X��9X��j��j��j��j�ě��ě��ě��ě���/��/��/��h�������������o�+�C��\)�\)��P��w��w��w�#�
�#�
�',1�8Q�8Q�<j�@��@��aG��e`B�e`B�m�h�u�y�#��o��C���C���C����罩�罩����X[hmtv���xthd[VSXXXX�������������������������������������{! @BN[dt������tgVJC@?@�������$#���������������������������������������������������������������������
#0UbjrvsD<20+

��������������������#./<@HUaUQH</#aabntz����zncaaaaaaa��	
#%#$#
�����Y[gt}ytg[XYYYYYYYYYY}������������������}#/<HMLD@</#
)).4))))))))))�����

��������")B[abg{��slg[J5"NOQZ[hrt���{tih[UONN���������������������������������FRYanz�������vnaUGEF[afmz�����������zn[[QT`ahmmooma\UTQNQQQQ26?BO[hqlhd^[OBA6422�����������������~~����������������������������������������������������u��������������vuuuu)+5654)$#/BHUay���zmaUH1oz������������zmfabo������������������������������������������������������������������������������������������������������������������������������

��������)))6B@6)'())))))))))CO[bc[XOJCCCCCCCCCCCBNft~���������t[N?;BR[_gtw������~tmg[WRR

�����+-'�������dmz��������zmaYVUV\d��������������������66BGMOOPQONFB>656666<<IUY^VUQI<36;<<<<<<8CHTamnmiaTH;40/0/28xz���������������zyx�����������������}}���������������������in{���������{sqrmmoicgot��������tsgdcccc��������������������JOS[hiihfgd[WOEDJJJJ����#)?@9)��������BN[a[NI?4)���bght��������tiga`a^b)5BN_gmoopg[K:5 ��������~�����������������������������������������	
#%+-0.#
���#)./7<HNTUPHF</#()-6BBO[chjjh`[OB6)(�������������������ڿ��������ÿĿѿѿӿݿ޿�ݿؿѿĿ��������	�����	������	�	�	�	�	�	�	�	�	�	����������������������������������@�=�3�-�*�3�@�B�I�A�@�@�@�@�@�@�@�@�@�@�ʾȾľ̾׾߾�����	�����	�����׾ʿ	���׾��������ʾ׾��;�G�Q�H�:�4��	�h�^�^�h�t�~āăćā�w�t�h�h�h�h�h�h�h�h�U�U�H�C�<�;�8�<�B�H�U�a�g�i�b�a�a�U�U�U�����������Ľнݽ�����������ݽнĽ����������t�p�v�������A�Z�f�Z�4���ѽн��z�q�n�j�i�n�n�yÇÍÓÖàâãëàÓÇ�z�H�?�<�;�8�<�H�O�U�\�a�i�r�Å�z�n�a�U�H�m�j�m�r�m�i�m�s�z�~�����z�q�m�m�m�m�m�mìèàÓÒÎÓÛàìùúýúùòìììì�f�c�a�a�f�s�u�x�w�s�f�f�f�f�f�f�f�f�f�f�Z�M�A�4�(�������(�4�M�X�]�Z�Z�a�Z�O�B�6�*�'�*�*�.�6�B�O�[�c�g�_�h�h�b�[�O���������������������������������������������������
���*�9�F�P�T�Q�O�6�*���������������*�C�O�\�h�|�y�h�]�H�6��뼘������~�}�������������������������������������	��"�.�;�@�L�R�X�T�;�.�"�ǔǌǈǃǈǔǡǥǭǡǔǔǔǔǔǔǔǔǔǔ���g�Z�J�<�;�A�N�d�g���������������������/�*�	�������������������	�"�;�@�F�E�;�/�����������	���"�%�*�"���	����������.�)�"��!� ��"�.�8�;�G�M�I�H�G�=�;�.�.��׾ʾ����������žʾ���� ��
��	����������������$��������������������|�~��������������ƾ��������������g�c�d�g�s�v�������s�g�g�g�g�g�g�g�g�g�g�׾̾ʾɾž¾ʾ׾��������׾׾׾׿���������������������������������������E�E�E�E�E�E�E�FFF#FJFnF�F�F�F�FVFE�E�Ǝ�x�i�c�f�tƁƎƧƳ����������������ƳƎ����������������
��$�0�3�7�7�2�0�$���(�������(�5�A�N�R�Z�Z�Z�N�I�A�5�(���������u�m�g�m�y�����������Ŀÿ��������G�C�;�6�1�7�;�@�G�J�O�T�U�T�G�G�G�G�G�Gìçèìù��������ùìììììììììì������3�@�Y�e�r�{�����������r�Y�3��ܿѿĿÿ����¿Ŀѿݿ���������������������������������������������������������������������������������������������������������������������%�(� ����������������������}�������������������������$��!�$�0�=�=�=�6�0�$�$�$�$�$�$�$�$�$�$�����������ʼ���.�7�=�<�3�%��ּ�����ĚĐėĩĿ����������������������ĳĚ�����������������������������������������S�S�N�S�_�i�l�x�������x�v�l�`�_�S�S�S�S�����������������������������������������A�5�1�-�*�5�N�Z�g�{�����������s�g�Z�N�A���������������ѿݿ����������ݿѿſĿ��)�������������)�6�B�N�V�T�O�J�B�6�)������������������������������������������&�M�Y�f�m�r�����r�f�Y�@����Ň�}�{�y�q�v�{ŇŔŗŠťšŠŘŔŇŇŇŇ���������������$�%�,�0�3�2�0�,�$�����Ϲȹù��¹ùϹչܹ����������ܹϹϹϹϽl�K�G�0�*�5�l�x���Ľнݽ�սĽ��������lÓÇ�z�n�Z�P�F�D�H�P�d�zÓàò÷÷ìàÓ¿³²¦ ¦²¿���������	������������¿ĦĚğĢġĢĦĮĿ����������������ĿĳĦ���������
��"���
�����������������������x�l�R�F�S�_�x�����ûлӻ׻лû����������������������!�-�9�-�!�����EEEEE EEEE*E7ECEPEXEVEPEMECE7E*ED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùõìçààÙÙàçìùÿ����������úù�M�B�H�M�Y�f�o�r�t�r�f�Y�M�M�M�M�M�M�M�M C m S P d L D I < W 8 ; k < X B 8 W 6 : S A P - S ] N = J ' = _ I [ . = > J t > G - I 7 Y w K h B d s F J 7 E Y \ ' 1 ; S c e v > � _ @ H [ L    �  ;  e  M  d  H  �    �  �  {  �  }  �  `  +  �  a  =    �  �  >  �  Y  �  *  �  �  N    "  H  �  #  �  B    �  P  I  �  9  X  �    ,  v    �  �  �  ~  �  �  _  �  �  z  �  '  s  �  �  I  �  �  �  �  P  `;�o��o��t��#�
��j�D���T�����ͼ�����{��`B�o��o�ě���t��C��'����L�ͽ�C���P�0 ŽC��ixսD�������,1��`B�D����h�o���ě��y�#�aG��]/�'t���w�����y�#�#�
�'�7L�8Q�0 Ž�^5��Q�,1�H�9�<j��hs���T���T��\)�y�#��o��7L��7L��S�������\)��9X��hs�Ƨ�t��+������ͽ���B�2B!�jBg�BFB	O�BwVB��B �[B"w�B&|�Bw�BIB�BNB	R�B��B��B�B�mB�^B��Bj~BBB��B2A�I�B˕B�bB*:�B�/B�B3�B��BQ�B �OB�IB�,B*>�B,r�B�Bc�B�B�>B��B	��B	�5B��B-�(A�M�BWB=�B&��A�B��B4;B-B(��B
fB�3B.�B+�Bf�B	�B��B
��B�IB4B1~B�*B�8Bj�B�tB!�IBRBBJB	{aB��B��B ��B"��B&�"BJXB.`B�B?�B	5ZB�^B��B5�B2�BF)B�
BGWB=�B��B@[A��hB>�B�B*G2B��B��B4�B��BUtB E�BB-B��B)��B,A�B��BA�B�zB��B��B	}�B	�B��B.9XA��`B�!B@+B&�mA��5B��B?B��B)�B	��B?�B>4B�DBHwB	�;BZB
��B��BBdB?�B��B�xBC9Ay��A��DA�k�?��BAW?�AW��A�6�A�W�A)uA*�tA�{	A��A��A�D�AA�jA8�bA���A�EsA��A���@��2A_5"Be�A�gFA��*A�~cAaA�ATW:A��AI.qA�WAS��Ap�C���B��B	�A��!Aq�"Ad�=A�C�?ǰIA}`�A��:A�0$A�%A���B
�A��A���A�xE@�o6A�}A�.�A{+AևhB5�@���A���B	",>ڳ7A��A�7A��A���A聁@��L@d�tC���C�+A��@�0vAysA���A��J?�m�AY
AT&�A�rA�Q A)A(uA�rAŀYA�� A�pABS�A9 tA�e�A��A���A��)@�A`�hB?�A���A�\A���Aa=-AT��A���AH�A�R�AR�YAq$1C��<B�B	?�A��}As/Ad��A�|@?���A}��A��WA��A��YA���B	�A"�A�IA�N�@�+ A�N�A�ntAz��A�hB?^@��eA�x0B	?�>�U�A��A��pA���A�,A�lI@��S@\	cC��C��À
@�p                  *            L            
                  /            #                              C                     /                     5   3      	         %   $         
         6   (   
               2   $                        1            ;                              -            !   %                           ?   !                  %            )         1   !                        %            /   %            !                                 )            +                              !                                          =   !                  #            #         1                           %            '                              N���M��DNZ�NtOl'uP��NXx
N�s�O-�O�0/N�R�O =�N'��N���N+�!O:�"O%�XN7�aOu��O��N��6O#�N3�iO�H�O�)GN���N�2O3�N�6N��M�N�N�irN�wPq�*O�OVN�N�iO#��NB��N4H�Oۘ�O&J�N-(N6K�O�?	N��N �*Px�O��	N�e�N??iN��-O��N��qO4i�NB�O��:Nδ�O1��N�TzO�%�O���O/�DO�eNߤO�_N�VN�dO"��N��@NJ�  -  �  �  �  �  7  �  �     �  �    �  �  �  �  f    �  �  �  �  �  N  �  7  �  D  }  �  �  �  &  
m  �  �  �  �  �    �  �  :    {    d  �  �  L  �  �  �  @  �  �  �  *      �  C  g  �  �  `    0  	  A  �<49X��o�#�
��o�o�e`B��`B�o�T����w�u��C��e`B��o��o���㼴9X���
��j�\)�ě���/��j�����ě��ě���/�ě��o��/��/��h�C��o�o�������o�\)���\)�\)�'�w��w��w�<j�#�
�',1�8Q�ixսP�`�@��@��aG��e`B�e`B��+����y�#��o��C����㽋C�������{������X[hmtv���xthd[VSXXXX����������������������������������������! AEN[bgt������tgXKEAA�������
�������������������������������������������������������������������#0IU_dffdb^UI>6' #��������������������#/<BGHMH<<0/##aabntz����zncaaaaaaa��	
#$#"###
���Y[gt}ytg[XYYYYYYYYYY��������������������	
 #/<HIHC></#
	)).4))))))))))�����

�������� )5BNS^kjg[NB5)!R[`hlt���wtha[ZPRRRR�����������������������������������QV]anz�������znaUNMQnty������������znllnQT`ahmmooma\UTQNQQQQ26?BO[hqlhd^[OBA6422������������������������������������������������������������������������u��������������vuuuu)+5654)$#/>FUaz���znaUH/oz�����������zmibcgo�����������������������������������������������������������������������������������������������������������������������������

��������)))6B@6)'())))))))))CO[bc[XOJCCCCCCCCCCC?BNtx��������tg[PC@?R[_gtw������~tmg[WRR

�����+-'�������aejmz������zma^YXY[a��������������������66BGMOOPQONFB>656666<<IUY^VUQI<36;<<<<<<8CHTamnmiaTH;40/0/28���������������������������������������������������������in{���������{sqrmmoicgot��������tsgdcccc��������������������JOS[hiihfgd[WOEDJJJJ��� ,9:7)������)BJFB;0) ����bght��������tiga`a^b)5BN_gmoopg[K:5 ��������~����������������������������������������
##')&#
 #%./8<EHMSSOHD</# ()-6BBO[chjjh`[OB6)(�������������������ڿ��������ÿĿѿѿӿݿ޿�ݿؿѿĿ��������	�����	������	�	�	�	�	�	�	�	�	�	�������������������������������������@�=�3�-�*�3�@�B�I�A�@�@�@�@�@�@�@�@�@�@�׾ʾƾξ׾۾������	�����	�����׿	����׾ľ��������ʾ׾����"�.�;�?�"�	�h�^�^�h�t�~āăćā�w�t�h�h�h�h�h�h�h�h�H�D�<�<�:�<�H�I�U�a�f�h�a�a�`�U�H�H�H�H�������������Ľнݽ��������߽ݽнĽ����������������ݽ����(�2�8�(��н��������z�u�n�m�m�n�u�zÇÐÓßßÞÓÇ�z�z�z�z�H�A�?�A�H�U�W�a�e�n�n�z�{�z�z�n�a�U�H�H�m�j�m�r�m�i�m�s�z�~�����z�q�m�m�m�m�m�mìêàÝÓÐÓÝàèìõùùüùøïìì�f�c�a�a�f�s�u�x�w�s�f�f�f�f�f�f�f�f�f�f�4�(���������(�4�>�M�U�Y�U�M�A�4�O�M�B�6�/�)�)�+�-�6�B�D�O�[�`�`�[�d�[�O������������������������������������������������������*�6�C�M�P�O�C�6�*�������������C�O�S�\�^�Z�S�J�C�6�*������~�����������������������������������������	��"�.�5�;�D�L�G�F�;�.�"�ǔǌǈǃǈǔǡǥǭǡǔǔǔǔǔǔǔǔǔǔ�s�g�Z�P�A�A�N�Z�g�s�������������������s�	���������������������	��"�.�2�'�"��	�����������	���"�%�*�"���	����������.�)�"��!� ��"�.�8�;�G�M�I�H�G�=�;�.�.����׾ʾ����������ʾ׾�����������������������$������������������������������������������������������g�c�d�g�s�v�������s�g�g�g�g�g�g�g�g�g�g�׾̾ʾɾž¾ʾ׾��������׾׾׾׿���������������������������������������E�E�E�E�E�E�FFFF1FJFjF�F�F�F�FPFE�E�Ǝ�y�j�e�gƁƎƧƳ����������������ƳƧƎ����������������$�0�2�6�7�4�1�0�$���5�4�(�����(�5�A�F�I�A�6�5�5�5�5�5�5���������u�m�g�m�y�����������Ŀÿ��������G�C�;�6�1�7�;�@�G�J�O�T�U�T�G�G�G�G�G�Gìçèìù��������ùìììììììììì������'�3�L�Y�b�r�������~�r�[�@�3��ѿƿĿ����Ŀǿѿݿ������������ݿѿ�������������������������������������������������������������������������������������������������������!�$�"�����������������������}�������������������������$��!�$�0�=�=�=�6�0�$�$�$�$�$�$�$�$�$�$�����������ʼ���.�7�=�<�3�%��ּ�����ĭĦĚĖĜĦĬĿ�����������������Ŀĭ�����������������������������������������S�S�N�S�_�i�l�x�������x�v�l�`�_�S�S�S�S�����������������������������������������A�5�1�-�*�5�N�Z�g�{�����������s�g�Z�N�A�Ŀ����������Ŀɿѿݿ�����ݿѿοĿ��6�)����������)�6�B�J�O�R�P�D�B�6������������������������������������������&�M�Y�f�m�r�����r�f�Y�@����Ň�}�{�y�q�v�{ŇŔŗŠťšŠŘŔŇŇŇŇ���������������$�%�,�0�3�2�0�,�$�����Ϲȹù��¹ùϹչܹ����������ܹϹϹϹϽS�G�A�;�E�c�l�������ĽҽԽ̽Ľ������l�S�z�p�n�]�Z�V�a�zÅÓàêîññìàÓÇ�z¿³²¦ ¦²¿���������	������������¿ĦĚğĢġĢĦĮĿ����������������ĿĳĦ���������
��"���
�����������������������|�z�������������ûǻлѻʻû������������������������!�-�9�-�!�����EEEEE)E*E7ECEPEREREPEFECE7E*EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ùõìçààÙÙàçìùÿ����������úù�M�B�H�M�Y�f�o�r�t�r�f�Y�M�M�M�M�M�M�M�M C m T P d X D E : c * " k B X . 8 W . / R : P . L ] N 2 J  = _ I ^ . < 3 J t > ? " I 7 Y w K h 8 d s F J = 9 Y \ ' 1 ; M A e v > [ _ ? ? [ L    �  ;  q  M  !  l  �  �  x  �      }  �  `  �  s  a  �  �  �  x  >  �  H  �  *  x  �  �    "  H  �  �  �  �    �  P  �  \  9  X  X    ,  v  �  �  �  �  ~    �  _  �  �  z  �  v  b  �  �  I  G  �  �  p  P  `  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  E'  -  '  !            �  �  �  �  �  �  �  �  �      2  �  �  �  �  �  �  �  �  �  �  �  �    n  Q  4    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  S  &  �  |    �  �  �  �  �  �  �  �  �  v  j  ]  P  B  2  #      �  �  �  �  �  �  �  �  r  Z  z  z  g  R  :    �  �  �  m    �    %  (  4  %    �  �  �  �  �  W    �  �  �  t    �  �  �  �  �  �  �  �  �  �  z  q  g  [  O  :    �  �  �  a  .  �  �  �  �  �  �  �  �  �  �  �  q  d  a  ^  [  X  S  G                       �  �  �  �  r  N  (    �  �  �  �  �  F  �  �  �  �  �  �  �  �  I  �  �  �  '  �  �  t   �  �  �  �  �  �  �  �  �  �  �  �  �  �  S    �  �  �  t  _  �  �  �                	    �  �  �  �  �  ]  5  L  �  }  x  s  o  j  e  `  \  W  P  G  ?  6  -  $      
    �  �  �  �  �  �  �  �  �  �  �  �  z  _  E  *    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  j  �  �  �  �  �  �  �  �  �  �  h  D    �  �  j  ,  �  �  A  X  b  e  \  H  .        �  �  �  �  O    �  �    ]            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  K    �  �  {  1  �  p    �  �  �    W  �  �  �  �  �  �  �  o  Z  A  !  �  �  !  t  �  �  �  �  �  �  b  7    �  �  �  q  N  )    �  �  �  S    �  k  p  y  �  �  �  v  i  Y  F  .    �  �  �  _  ,  �  �  �  �  �  w  ]  B  &  
  �  �  �  �  d  @  (  
  �  �  L  	  �  �  "  :  G  N  L  C  6  '    �  �  �  �    P  �  �    �  X  �  �  �  �  �  �  �  �  �  v  b  I  '  �  �  y     g   �  7  .  $      �  �  �  �  �  �  �  r  Y  :    �  �  �  ^  �  �  �  �  �  ~  f  F  $    �  �  �  �  �  �  b  ;     �  /  6  @  D  =  1       �  �  �  �  m  >    �  �  s  G    }  z  w  t  q  m  g  b  \  W  O  E  ;  2  (         �   �  ]  �  �  �  �  �  �  �  �  g  E  !  �  �  �  q  2  �  G  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  i  �  �  �  �  �  �  �  �  ~  q  d  W  H  8  '       �   �   �  &        
    �  �  �  �  �  �  �  �  �  s  _  J  6  !  
V  
l  
f  
O  
  	�  	[  �  /  �  h    �  �  s  �  4  �  D  �  �  �  �  }  k  Z  K  @  7  ,      �  �  �  F  �  t      �  �  �  �  `  C  '  	  �  �  �  `  /  �  �    �    }   �  �  %  Y  v  �  �  �  �  �  v  c  I  #  �  �  h    �  1  �  �  �  �  �  �  �  �  p  ^  L  :  '    �  �  �  �  �  _  5  �  �  �  �  �  �  z  o  c  U  G  9  %    �  �  �  �  �  g          �  �  �  �  �  �  �  �  d  @    �  �  �  s  G  �  �  �  �  �  �  N    �  �  �  V    �  l  �  �    N  _  {  �  �  �  �  �  �  y  `  :    �  �  <  �  �  8  �  w  2  :  2  )  !    '  2  <  G  Q  [  e  l  p  u  y  }  �  �  �    
      �  �  �  �  �  �  �  �  �  �    l  Y  D  /    K  h  x  z  w  t  m  _  Q  O  O  D  ,    �  �  @  �  Y  �      �  �  �  �  �  �  �  �  �  �  s  \  >  !    U  �  �  d  Y  O  D  :  0  %        �  �  �  �  �  �  �  �  q  X  �  �  �  �  �  p  O  #  �  �  �  X  $  �  �  F  �  3  _   �  N  o  �  �  �  t  X  (  �  �  A  �  k  �  x  �  T  �  �  Q  L  J  G  D  B  ?  =  :  8  5  1  ,  '  !              �  �  �  �  �  �  �  U  %  �  �  �  T  "  �  �  d    �  k  �  �  �  }  t  k  a  W  M  C  6  &       �   �   �   �   �   �  �  �  �  �  �  }  ]  ;    �  �  �  S    �  �  7  �  �    ,  �  �    (  ;  @  =  /      �  �  �  D  �  R  �  w  (  �  �  �  �  �  �  {  ^  >    �  �  x  4  �  \  �    J  A  �  �    �  )  3  &    	  �  �  �  �  �  �  {  b  G  +    �  �  �  {  o  k  i  g  g  c  [  O  >  ,      �  �  �  a  *  &  "        �  �  �  �  �  �  _  E  8  0  6  =  E  N      �  �  �  �  �  i  D    �  �  �  {  S  $  �  �  �  �    �  �  �  �  �  �  [  4  	  �  �  
  �  �  ~  H    �  �  �  �  �  �  �  �  �  �  �  o  G    �  ]    �  <  �  �  �  �    @  ;  *    �  �  �  �    D  �  �  0  �     �    �  g  X  I  9  )       �  �  �  �  �  j  D    �  �  ^     �  �  �  �  �  l  >  	  �  �  A  �  �  )  �  T  �  �  '  r  �  �  �  �  �  �  �  �  �  �  {  o  b  U  G  6  &       �   �    Q  1  +  *  E  L  /    �  �  j  7    �  �  �  V  �  +      �  �  �  �  �  �  �  �  �  �  }  c  I  .     �   �   �  
�  
�  
    "  ,  /  &  
  
�  
�  
1  	�  	.  �  �  �  �  �  h  �  	  �  �  �  �  �  �  U    �  �  E  �  �    �  *  �  �  A    �  �  w  N  '    �  �  �  y  F    �     �  ;  �    �  �  �  �  �  �  �  �  �  l  T  9    �  �  �  �  {  V  1