CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��G�z�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�c�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���w   max       =C�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F\(��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vmG�z�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�]�           7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��P   max       <�h       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�G�   max       B4��       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�}(   max       B4��       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�v�   max       C�9�       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >JbW   max       C�G�       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�ݗ�+j�   max       ?ҙ0��)       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��-   max       =o       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F\(��     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vl�\)     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̝        max       @�A`           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?ҙ0��)     �  ]            	      0   0   =               3   "      2                  '               @                   i      h   
      $      
      $      !            :         	   
               0   \                        &      ,   (   N���N��N�	gN�P]N<��P�\P�c�PR��N�O�N�>N��NwɮPc�O�OEN�S%O��N2�O@weN�O �N:��O���N�IxN��=O1,^On�}P$>�N`,O�� Oߦ�OS�O@�QP?��N���P8O"!JO�J�O��FN9�O��O�+�O�m�Nj�}O�6fN�N��M�L�O��AO��O��N��IN���O�O��N/�cOe	FO���Pa��N깲N&�`M��N��DO�2�N���N��OM"_N+�O�k�O��NV��=C�<e`B%@  %@  ��o���
�ě��t��49X�49X�D���T���T���e`B��o���㼬1��9X��j��j��j�ě��ě����ͼ�����������������/��/��/��`B�����o�\)�������#�
�#�
�#�
�49X�8Q�8Q�<j�<j�H�9�H�9�H�9�L�ͽL�ͽP�`�T���T���T���aG��e`B�e`B�e`B�e`B�y�#��%��%�����O߽�hs���w���w���w)5ABLMB5)&#/045771/#��������������������uz�������������zuuuu8BOV[[[SOMB=88888888j{������� ������{tlj���#b������n<0������hO;*('+6BJ\[t�����JN[cgjlmjg[NNJHDJJJJyz������������~zyyyyyz�����������zwvyyyy,58BLNPSPNEB5/,,,,,,BNg�����������ypWBEBCOnz���������aUFJIACz�����������zyusszzz��������������������7;BHIHH<;;1077777777��������������������	
		
#$/<H=<0/#
��������������������N[g�����������tg\VEN��������������������5;HJQTYXTH;412555555_ajmz�����zmka_\]_��������������������)6EX[h�������hOFI8()����������������������������������������������������������������������������������������������������AHUz�������zaZIDB>>A�������������������������594��������?IUbeggihibUIE=?@=??�����
)/2,&�������
#%%"
�������#(/2<=B<5/-#15BFNZ[grrod[QNB<5-1KRg���������mg[NMPLKz������������������z����������������������������������y{���������{usqryyyy))-5@BNRQNLB85)������������������������������������������������	���������#<IUbb_aaUG>:6-#ptv����������tppppppst������������|vtsss��������������������[`gt��������|sg[XVW[���������������������
#09=90)#
������<FUanz������zunaUH<<����������������������������������������������

 ��������������������������������$)5BFNQV[ag[NB5) $ht{�����������{thhhh36BBJMB6403333333333~����������������|w~�����������������
$<HLLH<4/,# 
���#'/6;?@?9/(#���������������������V�S�K�O�V�W�b�o�r�s�r�o�b�^�V�V�V�V�V�VŹŲŭŸŹ������������� ����������ŻŹŹ�4�1�0�4�8�A�G�M�Z�a�d�Z�M�A�4�4�4�4�4�4�����������������������	�
������a�W�Z�a�n�n�n�zÆ�z�s�n�a�a�a�a�a�a�a�a�ѿ����y�f�^�`�z�����ѿ���(�@�A�.��������������x�r�U�_�������������
��	����ݿڿ˿����m�T�;�"���.�;�G�b�q�y���ĿݾA�=�7�A�G�M�Z�f�s�u�}�v�s�f�Z�M�A�A�A�A���������������������������������������������������������������
�����������Ҿ����������������������������������������ɾ��¾��Ǿ׾��	�"�;�G�^�b�;�.������������������������
��#�,�-�*��
�������z�v�v�zÆÇÈÓàéìíîìàÓÇ�z�z�z�`�g�e�]�Q�;�.��	�����������.�G�T�`������(�)�5�5�5�(����������Y�L�I�L�V�Y�e�r�~���������������~�r�e�Y�	�������������	�����	�	�	�	�	�	�	�	�ʾž¾����¾ľʾ׾׾��������׾�ìèäìöùû��������ùìììììììì�p�o�j�m�x�����������������������������p�������������������������������������������������������������������������������������ƻƳ�������������������������������������������$�0�:�9�4�0�'�$����ݹù������Ϲܺ�'�3�7�3�4�>�3�'������<�<�4�;�<�<�H�U�^�a�c�a�`�V�U�H�<�<�<�<�ݿѿĿ����������Ŀѿݿ������	������(����&�5�A�N�Z�g�s���������s�g�A�5�(�0�/�0�<�>�I�V�b�n�{łŅ�|�{�n�b�U�I�<�0�t�i�h�X�P�\�h�tāčđĚğĨĦĦĚčā�t�N�B�A�K�U�g�s���������������������s�g�N���������������������
���
�
���������������������ȼ���!�-�2�0�+�"�����ּ��-�"�$�*�-�:�F�S�_�l�o�o�t�l�_�S�F�:�-�-ĳĦĚčĄ�w�t�h�uāčĚĦĮĴĸķĿĿĳ�ÿ����������Ŀѿݿ����������ݿѿ���������(�3�5�5�5�/�(����������������ŹŹűŹ�������������������������������������������)�0�/�+�)���ŹŸŢŇŁŇŖŠŭŹ����������������žŹ¿»²¦ ¤¦²¿����������¿¿¿¿¿¿������׾;ɾ׾����"�.�5�7�.�"��	�������������»ûлջܻ��ܻлû����������6�3�+�)��������%�)�1�6�8�9�9�6�6�����x�w�x���������������������������������
����'�4�@�M�U�f�y�~�u�f�Y�4�'��нɽ����������������Ľʽѽ޽������ݽн������|�~���Ľн��"�(�����ݽͽĽ����<�3�/�&�&�+�/�0�<�H�K�H�H�@�<�<�<�<�<�<ƧƥƝƧƳ��������������������������ƳƧ�L�I�D�@�@�L�Y�e�r�v�~�z�r�h�e�Y�L�L�L�L���������������������	���#�-� ������������#�0�2�6�0�#����������� �����������"�'�3�=�;�6�4�'��� ���������������ùܹ�����������߹ù��Y�M�@�=�>�B�M�Y�r������ۼ���ּ���Y��������������!�.�/�/�.�!������������������������������������ֺӺɺƺɺɺʺֺۺ���ֺֺֺֺֺֺֺ��I�F�=�0�%�0�=�?�I�V�Z�b�h�b�V�P�I�I�I�I�n�j�i�k�n�u�zÇÙìù������úèÓÇ�z�nùòìëäâìù��������������þùùùù���}���������������������������������������������������������ľʾξѾо˾¾������;�7�7�;�F�H�K�Q�M�H�;�;�;�;�;�;�;�;�;�;EEEEE*EAEIEPE\EdEcEiE~EuEoEiERE7EEE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��!�����!�-�:�B�F�K�F�:�-�!�!�!�!�!�! i h 3 ] N 7 o T < ^ V U > 2 %  ; . ^ X  L \ 9 _ " 4 Q U , a C 3 G E 4 W , 1 c ^ " 6 N 4 � J ' i [ < � - 8 E ; ? - H T z V G d > D Y t E X    3  �  �  {  �    �    3  )  �  �  �      X  �  M  :  P  m  L  �  �  �  �  �  &  �  q  �  G  �  �  a  Z    �  b  �  �  �  6  �  3  %  e  �  e  �  h  H  �  =  �  �  �    +  9  �  @    ?  �  <  �  m  �<�h%@  ���
�o�D���Y��]/��O߼��㼓t��u��o��o�D���C���O߼��ͽ'��ͼ���h��%������P�]/��Q�C��8Q�q���'aG��	7L�\)�C��8Q콃o���P�8Q�L�ͽ}󶽙���H�9���w�H�9�Y��L�ͽ����������m�h�u��7L����e`B��O߽��`��P���P�u�u��7L��{���㽑hs��"ѽ��P���#�����{BB�iB��B@�B�B*�1B%�(B3�B�'BXUB G�B��B
�`B�,B P�B!CwA���B!1A�G�B�B��B
6B�1A�!�A�xBT�B�qB!�yB�.B�}B!B�wB��B4PB-� B'(yB��B�rB*NB_�B
�BgB°B��B) �B�HB �?B �B"ԻB&�lB
n�B
�RB��B	��B^B$�,BiBI�B�(B"�B#��B��B�B
��BB4��B��BJkB�B�pBMB��B�BJ	Bl�B*0�B&? B(bB��B��B ��B��B
A"B�ZB �kB!BhA�}�B!�A�}(ByZB�}B
 �B�`A��FA���B@�BJ�B!B�B��B�pB?�B�,B�sBJ�B-��B'WB��B|�BG4B2qB	B�B;	B�BF<B)'7B��B �'B ��B"@�B&��B
�DB
@B:�B	ǃBA�B$P�BǛB�!B?JB"��B#ĨBûB~#B
��B=1B4��B�iB@�B��BhB3�A��A<�A�s�A�B�Az��A���Aq%AA?LA�)pA�m(AJ!AZͫA��A�0A_�+A��V?�~sA��>AR��A�NVA�eDA��CA�LBG�B	#�?%7�A��WA{�]A�jA��Aܥ�A�jSA�
�A��@��]A��A}!A���A�[A�hA���A��`AZr@�/yAի�@��@�ZA(�xA(\$A�UB"�?ځ0A��A��@���>�v�@��BA	h@[2@=ljB99A��A�EA��aAL�A�[aC��C�9�@t��BCWA�Y�A<k�A���A�~�A|��A��HAq�UA@��AрA�9�AJ��AZA���AɄ�A_ A��?�.A��AR��A�tqA�}�A��3A���BQnB	@�?.B�AĀWA|�A�y"AA�~�A���A��A��@xl�A�RA|AA�QA�MYA���A��=A�`�A\�X@�r�A�n7@�X@�eLA(�A"B�A�v�B�\?�՗A���A�|�@�}�>JbW@��;A�{@^�@4��B�A�c&A�x�A��0AL��A�z�C���C�G�@q��            	      1   1   =   	            4   #      3                  (               A   	      !         i      i         %            $      "            :         	                  0   ]                        '      -   )                     ?   A   4               3   %      #                  %               -         !         )      +                  !                           +                     #   -                              #                        7   ?   !               %   %      #                  %               #                                                                     +                        -                                    NA��N��LN_u�N[N<��P��P��O�q�N�O�N�>N��NwɮO�UwO�OEN��oO�
�N2�O@weN�O �N:��O���N§�N��=O#hO"3�O���N`,OE9�O�MN��RN���O�N���O�'�O"!JOG�*O7�N9�O��O��=Oy�dNj�}O�6fN�NS�+M�L�O<r{O��O��N��IN���O�O�f�N/�cOM�fO�ʌPP&eN�V�N&�`M��N��DO�2�Nl�_N��OM"_N+�O�N��JNV��  3  �  �  �    ,  "    z  `  C  �  �    /  G  U  �  �  �  �  �  �  �  �  �    5  �  
  �  m    @  
�    �  �    }  �  8  �  �  �  �  k  	  �    �  5  �  u    o  �  Z    7  �  �  �  K  H  �  �  	�    �=o<T����o�o��o�49X�o�o�49X�49X�D���T����h�e`B����ě���1��9X��j��j��j�ě����ͼ��ͼ�/�o�,1�������o��`B��㽏\)��󶽁%�\)�0 Ž@����#�
�'D���49X�8Q�8Q�D���<j��C��H�9�H�9�L�ͽL�ͽP�`�e`B�T���Y��m�h�y�#�m�h�e`B�e`B�y�#��%��������O߽�hs��-��1���w#)58BDB@5))!#########//34760/#���������������������������������������8BOV[[[SOMB=88888888u���������������{pou���#b������nU<0����4;BO[ht{���{th[OI944JN[cgjlmjg[NNJHDJJJJyz������������~zyyyyyz�����������zwvyyyy,58BLNPSPNEB5/,,,,,,Y[et�������������c[YCOnz���������aUFJIACuz���������zxuuuuuuu��������������������7;BHIHH<;;1077777777��������������������	
		
#$/<H=<0/#
��������������������N[g�����������tg\VEN��������������������5;HJQTYXTH;412555555`aemz|}~����zma_]^`��������������������38AL[ht{�����t[XOJB3����������������������������������������������������������������������������������������������������OUanz�������znaUPLLO������������������������"&$��������?IUbeggihibUIE=?@=??����!)--(!�������

�������#(/2<=B<5/-#15BFNZ[grrod[QNB<5-1KSgt����������kg[QMK������������������������������������������������������y{���������{usqryyyy05BDNPPNIB5-00000000������������������������������������������������	���������#<IUbb_aaUG>:6-#ptv����������tppppppst������������|vtsss��������������������Y[dgt�������wmgb[XXY����������������������
#07;70'# 
�����GUanz�����|znaUHA==G�����������������������������������������

 ��������������������������������$)5BFNQV[ag[NB5) $u�����������uuuuuuuu36BBJMB6403333333333~����������������|w~����������������
#*/5;0#
 ��!#/7;<==<6/#!!!!���������������������V�R�T�V�`�b�k�o�q�o�n�b�V�V�V�V�V�V�V�VŹůŹź������������������������ŹŹŹŹ�A�7�4�2�4�<�A�E�M�Z�^�\�Z�M�A�A�A�A�A�A�������������������������������������a�W�Z�a�n�n�n�zÆ�z�s�n�a�a�a�a�a�a�a�a�����y�h�i�}�������Ŀѿ����5�3�#���ѿ����������z�u�\�d���������������	��	��m�`�Y�^�^�j�y���������Ŀɿ˿ȿ��������m�A�=�7�A�G�M�Z�f�s�u�}�v�s�f�Z�M�A�A�A�A���������������������������������������������������������������
�����������Ҿ��������������������������������������������׾ӾҾ׾������	��"�3�9�<��	���������������������
��#�,�-�*��
������Ç�}�z�z�x�zÇÓàãåæàÓÇÇÇÇÇÇ��	����������"�.�G�Z�b�_�X�K�;�.�������(�)�5�5�5�(����������Y�L�I�L�V�Y�e�r�~���������������~�r�e�Y�	�������������	�����	�	�	�	�	�	�	�	�ʾž¾����¾ľʾ׾׾��������׾�ìèäìöùû��������ùìììììììì�p�o�j�m�x�����������������������������p��������������������������������������������������������������������������������������ƽƹ��������������������������������������������$�0�6�5�0�0�$������ܹù������ùϹܹ���#�*�*�'������<�<�4�;�<�<�H�U�^�a�c�a�`�V�U�H�<�<�<�<�ѿĿ������������Ŀѿݿ��� ������ݿ��0�(�)�,�6�A�N�Z�g�s�����������s�g�N�A�0�I�?�A�I�U�X�b�n�{ŀń�{�{�n�b�U�I�I�I�I�t�s�h�`�]�h�o�t�yāčĎĎčā�w�t�t�t�t�g�^�W�V�Y�d�s�����������������������s�g���������������������
���
�
�����������ӼƼƼ˼ռ������#�&�&�#�������ӻ-�"�$�*�-�:�F�S�_�l�o�o�t�l�_�S�F�:�-�-Ěčĉā�|�t�o�t�yāčĚĦĪıĴĳĲĦĚ�Ŀ������Ŀѿݿ�������������ݿѿ���������(�3�5�5�5�/�(����������������ŹŹűŹ����������������������������������������������)�.�)���œŐŕŝŠŭŹ������������������ŹŭŠœ¿»²¦ ¤¦²¿����������¿¿¿¿¿¿������׾;ɾ׾����"�.�5�7�.�"��	�������������»ûлջܻ��ܻлû��������������
���!�)�/�4�)�������������x�w�x�������������������������������'�#����'�4�@�M�Y�f�n�r�h�f�Y�M�@�4�'�нɽ����������������Ľʽѽ޽������ݽн������|�~���Ľн��"�(�����ݽͽĽ����<�3�/�&�&�+�/�0�<�H�K�H�H�@�<�<�<�<�<�<ƧƥƝƧƳ��������������������������ƳƧ�L�I�D�@�@�L�Y�e�r�v�~�z�r�h�e�Y�L�L�L�L�������������������������#�(�#��
���������#�0�2�6�0�#����������������������� �'�0�4�;�9�4�'����������������ùܹ����������ݹϹù����Y�M�B�>�?�D�M�Y�r�����Ѽ׼��߼ּ���Y���������!�-�,�!�������������������������������������������ֺӺɺƺɺɺʺֺۺ���ֺֺֺֺֺֺֺ��I�F�=�0�%�0�=�?�I�V�Z�b�h�b�V�P�I�I�I�I�n�j�i�k�n�u�zÇÙìù������úèÓÇ�z�nùîìæäìù����������ùùùùùùùù���}���������������������������������������������������������ľʾξѾо˾¾������;�7�7�;�F�H�K�Q�M�H�;�;�;�;�;�;�;�;�;�;EEEE!E*E/E7E>ECEPE\EcE\E[EPENECE7E*EE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��!�����!�-�:�B�F�K�F�:�-�!�!�!�!�!�! ? f 6 J N 6 o 9 < ^ V U . 2 %  ; . ^ X  L Q 9 ^  ( Q V  M    G : 4 M * 1 c `  6 N 4 ^ J 0 i [ < � - 7 E : 2 - F T z V G X > D Y j D X  h    o  g  {    �  �    3  )  �    �  �  �  X  �  M  :  P  m  �  �  m  ^  �  �  �  o    �  �  �  �  a  �  �  �  b  �  �  �  6  �  l  %  �  �  e  �  h  H  V  =  �  Q  �  �  +  9  �  @  �  ?  �  <  w    �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�      #  *  0  1  2  3  3  0  .  ,  &      	  �  �  �  �  �  �  �  �  �  �  �  �  �  }  n  ^  L  8    �  �  �  ^  2  �  �  �  �  �  �  �  �  �  �  �  w  h  W  G  7       �   �  �  �  �  �  �  �  �  �  �  �  �  �  s  L  %  �  �  �  T      E  i  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    +  *        �  �  �  �  �  �  �  s  F  �  d  �   l    "      �  �  b    �  �  ?  �  �  /  �  �  E  �  P   �  �  �  A  �  �  �        �  �  �  c  &  �  �  0  �  �  �  z  y  x  u  p  k  a  X  G  5       �  �  �  e  4      �   �  `  R  D  7  +      �  �  �  �  �  e  ?        �   �   �   ~  C  1      �  �  �  �  �  �  �  �  �  �  �  r  a  Q  @  0  �  �  �  �  �  �  �  �  �    s  f  Y  F  (  	   �   �   �   �    8  x  �  �  �  �  �  �  �  t  V  Y  f  Y  8  �  �    W      �  �  �  �  �  �  �  �  �  �  `  %  �  �  #  �  N  �    (  (  .  *         �  �  �  �  �  m  /  �  �  [    �  <  C  F  <  +    
  �  �  �  �  �  y  O  %  �  �    �  W  U  F  7  (      �  �  �  �  �  �  u  ^  G  1       �   �  �  �  �  �  x  e  K  /    �  �  �  }  L    �  �  �  O    �  �  �  �  �  �  �  �  �  �  {  v  q  l  h  c  ^  Y  T  O  �  �  �  �  ~  p  c  X  L  ?  2  %      �  �  �  �  E    �  �  �  �  �  p  Y  ?  $    �  �  �  |  S  *    �  �  �  �  �  y  t  j  [  H  9  1  !    �  �  �  L  �  �    �  1  �  �  �  �  �  �  �  �  �  �  �  z  t  o  j  e  a  G  %    �  �  �  �  �  �  �  �  q  \  G  1       �   �   �   �   �   k  �  �  �  �  �  �  u  _  C  "  �  �  �  \  *  �  �  `     �  R  r  �  �  �  �  �  x  `  E  !  �  �  �  8  �  x  �  Z  �  �  e  �  �  	    �  �  �  W  #  
  �  �  N  �  }  .  �  �  5  6  7  .       �  �  �  �  �  �  g  @    �  �  �  U    �  �  �  �  �  �  �  �  �  �  x  ]  @     �  �  �  ^    �  �  �    
    �  �  �  �  �  �  �  Q    �  �  -  �  �  9  G  z  �  }  n  \  H  1    �  �  �  i  %  �  �  .  �  q    y  �  �  �  
  :  Y  l  \  ;    �  �  �  t  ?  �  �  �    	�  
7  
�  
�  
�  
�  
�  
�  
�  
�  
�  
+  	�  	  O  r    V  %  �  @  3  &         �  �  �  �  �  �  �  �  �  �  �  �  �  �  
  
�  
�  
�  
�  
�  
�  
�  
�  
�  
T  	�  	�  �  R  �  �  �  )  2    �  �  �  �  �  �  �  �    �  �  �  �  �  {  i  W  D  0  �  �  �  �  �  �  �  �  �  j  ?    �  �  Q  �  u    �  �  �  �  �  �  �  �  �  �  �  �  m  C    �  h  	  �  4  �  5      �  �  �  �  �  �  �  �  �  �  s  \  F  1      �  �  }  r  g  Z  L  9  '    �  �  �  �  �  �  �  x  j  [  @  "  y  �  �  �  �  x  h  W  3    �  �  k  e  N  V  7      �  �  �    .  7  5  ,      �  �  |  7  �  �  4  �  6  �  �  �  �  �  �  �  �  s  e  U  B  0      �  �  �  �  �  �  }  �  n  P  /    �  �  �  q  G    �  �  S  �  *  �    �  t  �  }  w  p  j  c  Z  Q  G  >  5  +  !       �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  z  o  a  R  D  6  '  �  ^  �  k  m  n  p  r  o  `  R  C  4       �  �  �  �  m  D    �  k  �  �  �  �  	  	  	  	   �  �  q    �  /  �  (  �  3  �  �  �  �  �  h  C    �  �  �  �  o  U    �  �  \  �  �      �  �  �  �  k  =    �  �  �  �  z  :    �  �  A  �  o  �  �  �  �  �  }  i  U  =  %    �  �  �  �  }  `  >    �  5  #    �  �  �  �  �  �  |  S  &  �  �  �  `  +   �   �   �  �  �  �  ~  j  T  9    �  �  �  �  g  3  �  �  t    q   �  f  m  s  u  q  f  S  >  $  �  �  �  E  �  �  (  �  @  �  %       �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  b  W  L  h  n  o  h  ^  S  E  3      �  �  �  �  �  �  m  V  c  �  �  �  �  �  x  U  )  �  �  |  -  �  s    �  ,  �    �  �  @  Z  C    
�  
�  
7  	�  	  �  @  �  �  �  $  �  �    �  �            �  �  �  �  �  |  c  C    �  �  3  �  ,  �  7  %      �  �  �  �  �  �  �  �  x  n  d  L  0    �  �  �  �      &  7  A  K  U  _  j  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  l  ^  P  @  0       �  �  �  �  �  �  �  v  _  G  ,    �  �  �  r  B    �  �  ;  �  �  &  �  (  :  F  K  K  F  <  /    	  �  �  �  �  [    �  �  E  �  H  6  $    �  �  �  �  �  �  |  g  R  >  ,    
        �  �  �  �  l  W  D  2    �  �  �  J  
  �  b  �  /  2    �  v  d  R  @  /      �  �  �  �  o  Q  4     �   �   �   �  �  �  	;  	t  	�  	�  	�  	�  	t  	?  �  �  Z  �  �    8    �  m  
�  
�      
�  
�  
�  
T  
  	�  	�  	F  �  �  N  �  �  .  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �