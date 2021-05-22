CDF       
      obs    J   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�;dZ�     (  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       PE�     (  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���m   max       <���     (  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?8Q��   max       @F�          �  !$   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v|�\)     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  8D   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�7�         (  8�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �1&�   max       <T��     (  :    latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Л   max       B4�     (  ;(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}�   max       B4B-     (  <P   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >F�   max       C��     (  =x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@P�   max       C���     (  >�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z     (  ?�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A     (  @�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A     (  B   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       PE�     (  C@   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�[W>�6z   max       ?���ڹ�Z     (  Dh   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���m   max       <�t�     (  E�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @F�z�G�     �  F�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v|�\)     �  RH   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            �  ]�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�7�         (  ^l   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�     (  _�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?��C�\��        `�               (         	      U            /                           
         $         -      z               (         A      #         &            0   +               
         	            	               7                        2P�N��N6�N�+�O�WMNTT�N$ŪNy��O�TPrAN�7LN�5�O ٳP�Nn̅OǲcN̂O�4O��O%ěN��OOs��Nz��No�	O��
PE�N�E-N���P*�O,|O�p9N}GN� �O�2HN$�-O�~N�]O��%P;�Nid�O��eN��NP�N#��N*�HO@%�O{��P.�vN��N�,9N�C�O�G>N�7�OY��N.��N`r�O���OC|�O))�N�O�O���O2 �O�+MOw�UO�L�O�K[NؑO��NC�~O��N=��N>��Oe��<���<���;�o��o�o�D���D����o���
���
�ě��#�
�#�
�#�
�49X�e`B��C���C���C���C���t����
��1��9X������/��h�������������o�t��t��t��t��t��t��#�
�#�
�,1�0 Ž0 Ž49X�8Q�8Q�8Q�8Q�<j�D���D���H�9�Y��]/�]/�aG��ixսu�u�u�����C���C���\)������
���罺^5�����F���#���m���,5PWUXF��������������������������� ���������������������������������ULH</#	��#/9HPUVU��

�����������������������������������
#()#
���������������������������0Ul~�����nbI?<0<BO[^d][OB;9<<<<<<<<��������������������������������������������#-0*#
��������
&)-)'���#*41, 
��������������������������������������.6COWYYSC6 ~������������������~@BN[gptyttg[NIB;@@@@ht�����������tha]__h##/<>G=<2/-)$#######ght������}tnh^gggggg���"*' 
�����������#5I{����wpvnI4
���PUacjifbaUQPJKPPPPPP`ablmz�����{zmga````�����������������������������������������1>HNMB5)�����./6<=@><4/-+........��������������������GIRTantxxxyxymaTKHFGjmz�{zvmhbjjjjjjjjjjz����������������{wzABGNPUX[b[RNBA;76;AAcgy�������������tlgc����������������������������������;HMTadmswz{uaH:4227;`adligba_\[]````````������������������������zn_Z`n����������)-25)]ht����tih]]]]]]]]]]����������������������
#/561&#
����������������������������������������������V[hqtytoh[RPVVVVVVVV��)4BNYgt�����tg[B.%5>BEIB<53)������ �����)+6;6.)"@BO[fhhhe[VOFB@@@@@@���+353(�����oty������������uuto:=EIUbkmnprrnbZUG><:abny{~}{{tnfbaa]\^aax������������{uqoprx,/7<HILUVYYWUH<2/,+,NTit�����������tgZTN������������������������������������"%)' �����~��������������}~~~~��������������������

�GHU^lnz��zsnaXULHGFGUUTQH?<;<AHNSUUUUUUU8<HQUKHD<88888888888�����tg[ZX[gt��������s�a�N�(����(�1�N�Z�g�s�������������s¦¤ ¦°²¼´²¦¦¦¦�ù¹��ùĹϹ׹ܹ�ܹҹϹùùùùùùù�Ɓ�ƁƌƆƌƎƚƧưƱƯƧƜƚƎƁƁƁƁ�����	���������ŹŭŦťŦŭŹ���������4�,�3�4�A�M�V�Y�M�A�4�4�4�4�4�4�4�4�4�4�����!�-�2�:�@�:�-�!���������t�p�t�v�t�h�l�t�x�t�t�t�t�t�t�
�����������������
������#�$�#��
�����t�s�������Ľ����:�A�D�A��ֽ������a�Z�X�`�a�n�z�}ÇÀ�z�n�a�a�a�a�a�a�a�a������������� ����������������߽�ݽϽ����������Ľнݽ�����������@�(�5�E�S�Z�f���������������������f�@�{�o�m�b�V�Q�Q�V�Z�b�j�o�{�{�{�{�{�{�{�{����ùìàÓÌÓìù���������������ÇÅ�z�x�z�~�}ÇÓÚàáããàÓÇÇÇÇ�f�Y�M�@�4�'��'�-�4�@�Y�f�p�r�w�z�r�k�f�"������׾Ҿ;վ���	��"�(�1�6�:�.�"���������������������������ʾξѾ̾�����������~����������������������������������������
���#�/�<�H�U�X�T�E�<�/�#�
���H�H�?�@�H�U�a�b�n�q�n�a�U�I�H�H�H�H�H�H�H�G�@�@�G�H�H�T�_�a�b�a�\�T�H�H�H�H�H�H�a�X�V�a�m�z������������������������z�a�����{�g�N�,�(�5�Z�i��������������������A�=�?�A�M�Z�f�s�v�y�s�f�Z�M�A�A�A�A�A�A�
�����������������
������
�
�
�
��������������I�b�{ņ��~�w�n�U�0��
�����������$�0�8�=�9�1�0�(�$����s�k�i�j�tāčĚĦĮĻ������ĿĴĦčā�s�[�W�O�J�O�[�h�p�t�u�t�h�[�[�[�[�[�[�[�[�m�f�a�m�u�y�����������������������y�m�m����ƧƟƕƚƧƳ�������������������������t�q�n�tāĆčđčā�t�t�t�t�t�t�t�t�t�t���������������)�C�O�h�l�h�O�B���������������������������������������������������������������������������������M�@�'����4�Y�r�����������ļ���������������������������������������������������������������	��"�/�7�'��	���������Z�Y�Z�g�s�����������s�g�Z�Z�Z�Z�Z�Z�Z�Z���ֺɺȺƺɺֺ���������������������������m�T�J�F�I�T�a�m�z���������l�i�i�l�x���������x�l�l�l�l�l�l�l�l�l�l�����|�}���������������������������������N�K�N�R�P�V�Z�g�s�������������z�s�g�Z�N�����������������������������������������#���0�L�e�����ֺ�������ݺɺ��r�L�3�#��������������!�-�7�:�>�<�:�/�-�!������������ùɹϹع׹Ϲù����������������ʾƾľʾ׾����������׾ʾʾʾʾʾʾ��ʾ¾þʾ۾۾��	���$�$�"���	��ŔœőŔśŠŧŭŹ����������ŹŹŭŢŠŔ�H�?�;�2�/�#��� �"�/�;�H�R�[�^�`�[�T�H���w�~�������������������������������������������������������������������������������ݽֽݽ����(�4�;�3�$������ŭŨŠŔŀ�x�{ŇŔŠŭŹ����������Źŵŭ�лû����������û̻лܻ��������ܻػлS�Q�G�S�V�_�l�x�y�������������x�l�_�S�S����������'�@�M�`�c�`�Y�M�@�4�'����FF E�E�E�E�E�E�E�FF$F1F5F=F:F1F,F$FF��������·¿�������
��/�<�H�T�K�H�/�
���V�L�D�>�9�0�$����$�=�V�b�m�t�u�o�b�V�������ǼҼ��������������ּ����!�����������!�:�G�S�]�]�_�P�G�:�!�C�8�6�*����*�-�6�C�C�O�Z�X�O�C�C�C�C����ĿĻĹĬħĲĿ����������������������D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�D������������������Ŀ̿ѿӿտѿϿĿ�������EuEwEuEiEgE\ETEPELECE>ECEPEVE\EiEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������������ùìáááàæìù���� @ ` ) H 8 @ N � D F % d } 8 5 X T E K 6 M 1 B T . r G * g c ' d ; M O S M R 0 J Z � V ? 4 l H 0 | 3 , ^ B S P G O @ V  L L > q f O 7 b % T K y $ ?  �  �  M  	  �  i  R  �  l  *  �    �  �  �    �  n  �  y  '  �  �  �  ;  n  �  �  �  s  �  0      +      �  5  �  �  �  H  �  B  �  �  �  A    �  �    ;  �  `  �    �  d  �  3  �  E  B  �  Q  �  �  I  \  �  K  컣�
<T��%   ��`B�,1�t���`B�T����t���Q켴9X�u���
�e`B��9X�49X��`B�t��D���+���ͽ�w�����ͽ0 Ž}��w��㽙���,1�������D���#�
����D���]/�����w���P�0 Ž<j���
�L�ͽ<j�u��j��9X�u�}�aG������q�����-�q���}󶽕���O߽�t���O߽��T�\��9X��{�������1���ͽ���+���C��1&�Be B'�B8�B(cB`By BTQB��B:B&��B�:BD5B"RjB#64B,�B�.Bl�B!B/�
B4�B��Br�B\Bc/B|B%�TB�A��B۾B�B�B��B+A�s�A���B[WB�Ba�BD�B 
�A�ЛA�Z�B"ȐB{�BBB�<B2�B��BzB�B��B��B�~B�BB�jBLhB
�hB''�B(�B)xwBoB
@wB�B-cKB�B
��B!HB�tBK�BZB&�B
;=B@"B>�B@�B?�B�BBkB@BJ6B�GB'@.B�B;�B"@�B#B?5B��BPB!BQB0=nB4B-B	/ABC�BB�B�	B��B%��BփA�s�B�B��B�fB��B*��A���A��iB��B��BµBB�B��A�}�A�lAB"��B��B�B?KBD�BABB%B@�B�UB`�B	@�B}B@B��B9^B=3B
��B'5EB(?�B)��BF`B
=B��B-��B��B
�&B�RB�B?�B@8B<�B
��A��.A�C�>��6BZ/A��EA;ZR@pv�A���A�V�A(vA�D�A��1A,=9AB&:B^�AϝWA�Y�@�_AY�`AL��Ap�A���Aŏ�A���A�d>A�h�A>��A��lA��{B	�BA��A�AouB<qA�hCA�m�A�YA��P@��@��A�i�A�j>@=��A���@�@.AorA���A��@0�@h3�>F�AT�AX�>A�C�A��'A�(�A�rA2::A�G;@��Q@���@���C��A�J+B^8A`#A1B qA��C�X^Avv�C��KC�@A��A��A� >�xB`FA��_A:�$@z��A���A��EA'�A�~�A��fA*�1AA�\BH[A�-�A�z~@؃MAY�AL��Aq�A�IGAŀ�A��A���A���A=oA�A�g�B	�4AޜAڇ�An�BW$Aݑ�A�h>A��/A��+@�˭@��pA�I�A�@�@<\A���@��Ao��A��WA�}i@(#@h�>@P�AS�pAY+fA��A�r�A��A�eA1��A��@��U@��X@�E�C���A¡B
�A�oAB �fA�yC�W�Axh�C��C�BA���                )         
      V            /                            
         $      	   .      z               )         B      #         &            0   ,                        	            
               7                        3   )                           9            )      '                              A         '                     '      #   /      %         +               7            #                                 )      !                                                      #            #      #                              A                              %      #   -      #         +               1            #                                 )                              Oj5NA�N6�N�͢O�WMNTT�N$ŪNy��O�TO�{~NݚN�5�O ٳOþNn̅O���N̂N��uO(a�N� N��OONw�Nz��No�	O��(PE�N�E-N���O7^�N�8�Oj�UN}GN� �O�2HN$�-O�:�N�]O��%P54Nid�O�vN��NP�N#��N*�HO@%�O!��P(�Nx�N<�*NG|�O�$XN�7�OY��N.��N`r�Ogh@OC|�Nv�N�O�OD��O2 �O�+MON�O���O�K[NؑO��NC�~O��N=��N>��Oe��  �  �      �  �  �    �  t  F  <  !  �  �  �  t  �  <    �  j  .  5  �  �  =  �  s  �  N  �  �  �  �  �  Y  k  .  �    c  �  !  �  S  �  �  �  �    2    *  �  !  �  �  �  x  �  �  V  g  �  ;  b  c  �    D  >  \  <e`B<�t�;�o�o�o�D���D����o���
�49X�D���#�
�#�
��j�49X��C���C���1��`B��9X��t���9X��1��9X��`B��/��h���H�9�o��+�����o�t����t��t���P�t��0 Ž#�
�,1�0 Ž0 Ž49X�8Q�e`B�<j�L�ͽL�ͽP�`�L�ͽH�9�Y��]/�]/�e`B�ixս���u��o�����C���\)���
������
���罺^5�����F���#���m
)*2:BA=5%����������������������� ���������������������������������ULH</#	��#/9HPUVU��

�����������������������������������
#()#
���������������������������.5<IU`fjmqrrmbURI?/.@BOT[\[ROKB?@@@@@@@@�����������������������������������������������
%$!
������
&)-)'���#/2/-)
��������������������������������������*6CHMNLD6/*��������������������@BN[gptyttg[NIB;@@@@bhit����������thc_ab##/<>G=<2/-)$#######ght������}tnh^gggggg���
'%
���������#5I{����wpvnI4
���PUacjifbaUQPJKPPPPPP`ablmz�����{zmga````�����������������������������������������!)0783)����./6<=@><4/-+........��������������������GIRTantxxxyxymaTKHFGjmz�{zvmhbjjjjjjjjjjxz����������������}xABGNPUX[b[RNBA;76;AAcgy�������������tlgc����������������������������������8;HTahmqtwwnaZH<7448`adligba_\[]````````������������������������zn_Z`n����������)-25)]ht����tih]]]]]]]]]]�����������������������	
#(02,#
�������������������������������������������Y[hltutkh[VSYYYYYYYY 25BNVgt�����tgNB4$$2%5>BEIB<53)������ �����)+6;6.)"@BO[fhhhe[VOFB@@@@@@��)120%������oty������������uutoDITUWab`UUUIGADDDDDDabny{~}{{tnfbaa]\^aa{~������������{xtsu{,/7<HILUVYYWUH<2/,+,NTit�����������tgZTN������������������������������������"%)' �����~��������������}~~~~��������������������

�GHU^lnz��zsnaXULHGFGUUTQH?<;<AHNSUUUUUUU8<HQUKHD<88888888888�����tg[ZX[gt��������g�Z�N�A�1�1�4�5�A�N�Z�g�s�w������t�s�g¦®²´²¦�ù¹��ùĹϹ׹ܹ�ܹҹϹùùùùùùù�ƚƐƎƈƍƎƚƧƭƯƬƧƚƚƚƚƚƚƚƚ�����	���������ŹŭŦťŦŭŹ���������4�,�3�4�A�M�V�Y�M�A�4�4�4�4�4�4�4�4�4�4�����!�-�2�:�@�:�-�!���������t�p�t�v�t�h�l�t�x�t�t�t�t�t�t�
�����������������
������#�$�#��
�����������������Ľнݽ��������ݽ����a�_�^�a�m�n�o�z�}�z�t�n�a�a�a�a�a�a�a�a������������� ����������������߽�ݽϽ����������Ľнݽ�����������f�Z�M�C�?�@�N�R�Z�f�s�������������s�f�{�o�m�b�V�Q�Q�V�Z�b�j�o�{�{�{�{�{�{�{�{����ùìàÔæìù��������� ���������ÇÅ�z�x�z�~�}ÇÓÚàáããàÓÇÇÇÇ�4�/�3�4�@�I�M�Y�f�i�r�r�s�r�f�Y�M�@�4�4��	������ݾܾ����	���"�$�%�"�������������������������þʾ˾ʾž�������������~��������������������������������
������������
��#�/�<�H�T�Q�C�<�/�#�
�H�H�?�@�H�U�a�b�n�q�n�a�U�I�H�H�H�H�H�H�H�G�@�@�G�H�H�T�_�a�b�a�\�T�H�H�H�H�H�H�a�Z�Y�^�a�m�z�����������������������z�a�����{�g�N�,�(�5�Z�i��������������������A�=�?�A�M�Z�f�s�v�y�s�f�Z�M�A�A�A�A�A�A�
�����������������
������
�
�
�
�����������
��#�0�<�G�G�<�0�,�#��
����������� �$�0�6�;�8�1�0�'�$����ā�t�s�p�r�t�zāčĚĦĳĹĹĴĳĦĚčā�[�W�O�J�O�[�h�p�t�u�t�h�[�[�[�[�[�[�[�[�m�f�a�m�u�y�����������������������y�m�m����ƧƟƕƚƧƳ�������������������������t�q�n�tāĆčđčā�t�t�t�t�t�t�t�t�t�t�������������������)�B�O�Z�e�[�B�������������������������������������������������������������������������������r�M�@�'����4�Y�r�������������������������������������������������������������������������	��"�/�4�/�%��	�������Z�Y�Z�g�s�����������s�g�Z�Z�Z�Z�Z�Z�Z�Z���ֺɺȺƺɺֺ���������������������������m�T�J�F�I�T�a�m�z���������l�i�i�l�x���������x�l�l�l�l�l�l�l�l�l�l�����|�}���������������������������������N�K�N�R�P�V�Z�g�s�������������z�s�g�Z�N�����������������������������������������3�'��1�L�e�����ɺ�������ܺɺ��r�L�3������!�-�3�-�-�!��������������������ùĹϹӹԹϹù����������������ʾɾȾʾ׾��������׾ʾʾʾʾʾʾʾʾ׾ѾǾžʾݾݾ��	���"�"����	����ŔœőŔśŠŧŭŹ����������ŹŹŭŢŠŔ�H�?�;�2�/�#��� �"�/�;�H�R�[�^�`�[�T�H���w�~������������������������������������������������������������������������������޽�����(�4�9�4�1�"��������ŭŨŠŔŀ�x�{ŇŔŠŭŹ����������Źŵŭ�û��������ûлܻ���޻ܻлûûûûûûS�Q�G�S�V�_�l�x�y�������������x�l�_�S�S�	���������'�4�@�O�Y�\�Y�F�@�4�'��	FF E�E�E�E�E�E�E�FF$F1F5F=F:F1F,F$FF��������·¿�������
��/�<�H�T�K�H�/�
���V�N�F�A�<�0�$��$�=�I�V�b�l�s�t�o�m�b�V�ʼ����ż̼׼��������������ּʽ!�����������!�:�G�S�]�]�_�P�G�:�!�C�8�6�*����*�-�6�C�C�O�Z�X�O�C�C�C�C����ĿĻĹĬħĲĿ����������������������D�D�D�D�EEEEEED�D�D�D�D�D�D�D�D�D������������������Ŀ̿ѿӿտѿϿĿ�������EuEwEuEiEgE\ETEPELECE>ECEPEVE\EiEuEuEuEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������������ùìáááàæìù���� % [ ) 5 8 @ N � D Z 8 d } = 5 V T : ? ) M 0 B T . r G * B \  d ; M O H M R 1 J T � V ? 4 l H  { 2 3 W C S P G O = V * L E > q Y > 7 b % T K y $ ?  �  `  M  �  �  i  R  �  l  �  1    �  �  �  �  �  �  y  �  '  �  �  �  �  n  �  �  �  I  �  0      +  6    �  
  �    �  H  �  B  �  �  X    {  P  y  �  ;  �  `  �  �  �  �  �  �  �  E  �    Q  �  �  I  \  �  K  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  D  m  ~  �  �  �  �  �  �  �  �  r  D    �  �  Q    �  ;  �  �  �  �  �  �  ~  y  r  k  a  T  G  :  *    
  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  o  V  =  #  �  �            �  �  �  �  �  �  �  �  �  �  g  J  -  �  �  �  �  v  V  6    �  �  k    �  W  �  �    ~  �  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  ~  �  �  �  �  �  �  �  �  �  �  �  }  s  i    ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R    �  b  �  �  �  �  }  h  Q  6    �  �  �  u  G    �  �  �  �  �  b  �  �  �  �    A  ]  q  s  j  P  '  �  �  3  m    \   �  �  �  �      4  B  E  :  &    �  e    �  �  /  �  �  #  <  8  4  /  -  0  3  6  4  ,  #        �  �  �  e     �  !        �  �  �  �  �  �  �  y  h  ^  A    �  �  K   �    D  d  |  �  �  �  v  V  3    �  �  >  �  �  3  �  S  �  �  �  �  �  �  �  �  �  h  C    �  �  �  k  ;  
  �  �  r  i  �  �  �  �  z  f  N  1    �  �  m  3  �  �  3  �  !  �  t  p  l  g  _  T  C  0    �  �  �  n  7  �  �  z  0   �   �  I  m  �  �  �  �  �  �  t  e  Q  ;     �  �  �  X  �  �  #  �  �      +  6  :  ;  3  #    �  �  �  �  C  �  �  $  B  �  �  �            �  �  �  �  �  e  7  
  �  �  �  `  �  �  �  �  �  x  k  \  L  6      �  �  �  \     �   �   ~  `  f  j  e  `  T  D  1      �  �  �  �  �  |  i  B    �  .  <  J  M  M  G  >  +    �  �  �  �  e  <    �  �  �  c  5  0  +  &             �  �  �  �  �  �  �  �  �  �  r  �  �  �  �  �  �  x  d  M  1  �  �  �  �  �  v  b  ]  y  �  �  �  �  }  Q  $     -    �  �  �  x  4  �  �  W  (  �    =  /          �  �  �  �  �  �  �  �  �  �  u  j  `  �    �  �  �  �  �  �  �  x  b  L  6  !    �  �  �  v  I     �  �  �  �  �  �      B  s  b  D    �  �  >  �  .  �    I  �  �  �  �  �  �  �  �  �  d  H  !  �  �  �  g  ,  �  �  @  �  s  �    6  L  H  '  �  r  �  R  �  �     �  _  	~  v  k  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  [  A  &    �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  ]  F  -     �  �  �  �  s  [  ?  !    �  �  �  z  Y  2    �  �  �  f  I  �  �  �  w  g  X  N  D  :  0  !    �  �  �  �  �  d  =    �  �  �  �  �  �  �  d  @  1    �  �  w  F  �  V  �  c  �  Y  =  $    �  �  �  �  �  w  n  }  r  f  `  N  8       �  k  _  K  3    �  �  �  �  }  [  :    �  �  �  �  ^  ;      $  	  �  �  �  _  1    �  �  �  L    �  e  �  %  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k  a  W  M  C            �  �  �  s  @    �  �  U    �  c  �  I  g  c  T  F  8  )      �  �  �  �  �  �  �  �  {  k  Z  J  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  !  �  �  �  k  ^  c  R  <  (    �  �  �  �  [  $  �    I  �  �  �  �  �  �  �                  �  �  �  �  �  �  S  K  B  9  1  (           �   �   �   �   �   �   �   �   �   �  �  ~  p  `  M  5    �  �  �  �  w  P  &  �  �  �  {  l  H  c  �  �  �  �  �  �  �  i  /  �  �  <  �  j  �  "  d  �  �  �  �  �  �  q  4  #      L  &  �  �  m  "  �  ]  �    '  {  �  �  �  �  �  �  �  �  �  �  �  �  �  p  :  �  �    �  �  �  �  �        �  �  �  �  �  �  P    �  �  O  
  �  x                  "  )  1  9  A  H  B  =  6  -  $                  �  �  �  �  �  �  �  �  �  z  2  �  �  m  U  *  $      �  �  �  �  �  �  a  :    �  �  �  i  J    �  �  �  �  �  �  h  Q  ?  *    �  �  �  O    �  �  -  �  Z  !      �  �  �  �  �  �  }  x  s  �  �  �  �  �  �  �  �  �  }  t  i  [  M  :  (    �  �  �  �  �  j  9    �  �  X  �  �  �  �  �  �  �  y  m  ^  L  7      �  �  �  z  A  	  �  �  �  �  p  Y  A  (    �  �  �  �  �  �  �  �  ^  ?  '  8  B  M  [  e  m  u  x  x  p  g  \  N  >  *    �  �  !   �  �  �  �  �  �  �  �  �  �  }  h  M  +  	  �  �  �  q  P  /  �  �  �  �  �  �  �  �  �  �  �  s  M     �  �  �  X    �  V  I  G  8    �  �  c    �  �  ,  �  ^  �    &  �  V  �  g  ]  I  /      �  �  �  Z    �  �  a  E    �  �  o  �  �  �  �  �  �  �  �  �  �  y  M    �  �  m  )  �  �  A  �  �       8  :  4  -    �  �  @  �  �  Q  �  o  �    7    b  ^  U  K  6    �  �  �  B    �  �  �  �  n  =    �  �  c  \  U  N  G  @  ;  6  1  ,  $        �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  3    �  �  �  o  F    �  q      �  �  �  }  R  !  �  �  �  C  �  �  �  -  �  k    �  5  D  "  �  �  �  ~  I    �  �  �  �  e  3  �  �  A  �  �  ,  >  G  M  6    �  �  �  �  Y  ,  �  �  �  T    �  �  8  �  \  =    �  �  �  �  y  V  2    �  �  �  g    �  v    �    �  �  [    �  �  8  
�  
�  
  	  �    Q  �  
  �  !  �