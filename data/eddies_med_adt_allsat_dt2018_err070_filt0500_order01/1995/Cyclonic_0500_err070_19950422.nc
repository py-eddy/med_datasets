CDF       
      obs    O   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�5?|�i     <  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�A0   max       PT��     <  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��o   max       <�t�     <   $   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>h�\)   max       @F������     X  !`   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vy\(�     X  -�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @R            �  :   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�          <  :�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       :�o     <  ;�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B4��     <  =(   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B4�F     <  >d   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��   max       C���     <  ?�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >3�C   max       C��F     <  @�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ;     <  B   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3     <  CT   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     <  D�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�A0   max       PK4U     <  E�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�qu�!�S   max       ?˩*0U2b     <  G   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       <D��     <  HD   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>h�\)   max       @F������     X  I�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @vy\(�     X  U�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @R            �  b0   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @�_          <  b�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     <  d   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n   max       ?˥�S���     �  eH   "                                           $                                    
               '      !      	      ;                     $                  !                  .   %            #      
                                    
         (O��SO��Oe�OϚN�_P�OZ�@O�{UN���O�ڹO� ,O�H O�*�OI_O&[*N�E�O���P&�O��xO8<\OD�O���N�s!O0&�M�A0N�*zO-%lN�1N�NK�sN�|PT��N�ZKPA/�N>��O � N��]Oц�N'�O>DO�"�N}
[O6ؽN�M�Ou�+N|�Nm0Ov�O8LFOQ��Oz��O7N�9O.�N��O�j�P@�P;kP��N�reNxeO,�GN��iN��O��~M�%�O��/O7AON_B:N�	OV�bN��Nbc�O@�N��N��N�p�O:��O��<�t�<#�
;�`B;ě�;D��:�o:�o��o��o��o��o���
��`B��`B��`B��`B�o�49X�D���D���T���T����o��o��C���t����
��1��9X��9X��j��j��j�ě��ě����ͼ�/��`B��h���������+�C��C��C��\)�\)��P��P��P����w�#�
�#�
�''49X�8Q�H�9�L�ͽT���T���T���T���]/�]/�aG��aG��aG��q���q���u�u�y�#��%��o��o���������������������������������������������������������������������������������������������������������
��������������
#+-+,&#
����558@NSgrwzuog[NB;:45������ � ���������x|��������������zrtx����#/52*
�������#/<UWZYXTSKH</#RZamps}�����zmaTOOQR����������������������������������36<BGOV[a_[ROB963333��������������������):Q[t���xqum[B6amz��������~zumka_]a���
�������y~���������������sy*6COhsxrh\6*� ����������������
��������������������������������
 
�������7;HTageca\TRH;763477������
 ����������MO[]c][UOLMMMMMMMMMMU[`hotnh[\[UUUUUUUUU��������������������#0Qn{�����n<0
	����������������������5BKPPJ;)������#$/73/#ABKNOZ[bgjnng[PNFB?A��������������������������������������X[bhptyth[VUXXXXXXXX����		��������)4:DHMKB6)�.5BNONLMB51.........Z[hqt�������thf[RRUZ����������������������������������������GHSU^anpnla\USNHGGGGCO[hlnih_[OICCCCCCCC����
#',,*#
����/<HSUY[UH?</#����������������������������������������w{~����������|{xuuxw<BJOS[_f[YOB<<<<<<<<������

�����������������������������#0<AFFFA0#
�����5;8+&&+=?5)���
0JX[WXULI=0

y}�������������|y7<IUWVWVUOI=<7447777Zacnusna`\ZZZZZZZZZZBCEJLO[chmtqlkhd[OFB��������������������������������������������������������~�.0<>EIMUVUID<0......����������%01.)&������������������������#&*/059;/-#":<BUahpu|zpnaURH>;9:stz�����������}utqss��������������������)40)&����ABNR[^_[YPNBB>7<AAAA05BN[][Z[VNB95010000���������������������������������������������!�����������
���������
��#�/�<�H�L�T�L�H�<�/�#�������������������"� ����������������������� �	��"�*�&�"���������U�H�?�<�6�<�H�U�a�j�n�o�n�l�n�z�n�h�a�U�����������ʾоѾʾ�����������������������Ƽƽ��������$�0�5�4�7�3�0�$����������Z�T�A�:�5�,�+�3�5�N�Z�g�s�w�{�x�m�g�^�Z�A�4������(�4�M�f�s�y�}�~�p�f�Z�M�A�Z�Q�M�D�A�9�A�M�Z�f�s�����}�s�f�`�Z�ZìàÓ�z�n�f�j�n�zÇàì÷����������ùìàÓÌÈÐ×àìù����������������ùìà������������������������� �������������O�B�0�/�6�O�[�h�tāčĦīĦĞďā�t�h�O�{�t�n�g�g�t¥¦«­¦D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D��a�[�U�Q�U�Y�a�h�n�zÆ�z�z�t�n�h�a�a�a�a�M�C�E�K�K�F�H�S�Z�s��������������f�V�M�.���������	��.�T�m�������s�m�`�G�.�����������������	��"�/�7�/� ��	����������������������+�5�B�D�5�2�)������6�*������������*�6�C�O�S�U�O�C�6��	���������	��"�.�1�3�7�7�4�.�"������������������������������������������������������������Ŀѿֿٿݿ��ݿѿĿ�����������������������������������������������&�)�6�B�C�B�7�6�.�)������������(�5�:�A�L�N�W�S�N�A�5�(���ݿ׿ѿѿοѿ׿ݿ������ݿݿݿݿݿݻ��������ûлԻлʻû��������������������x�u�x�{�����������������x�x�x�x�x�x�x�xùñìàÝÛàãìù����������úùùùù����r�Z�B�<�\�s�������������������������� �������������(�-�(�������������������������������!� ����������нϽϽнڽݽ������ݽнннннннпĿÿ������Ŀſѿ׿ݿ������������ݿѿ��/�"��	���	��"�/�7�;�H�T�Y�T�T�H�;�/����������'�M�f�p�}�������f�Y�4���Y�S�Y�\�e�p�r�r�t�x�r�e�Y�Y�Y�Y�Y�Y�Y�Y��ݺ�������!�-�4�-�%�!���������k�h�q���������������������������������k�������"�.�8�2�.�"�������������������������������������������������@�@�8�@�H�L�W�Y�e�f�l�e�c�Y�L�@�@�@�@�@�t�h�[�S�O�G�O�]�h�tĎĦĭıĦĚčĉā�t�׾Ӿʾɾʾ˾־׾ؾ������޾׾׾׾��������������������������������������������z�m�i�_�b�g�m�z���������������������������������������������������������������������������#�0�=�H�0�#��
������������������������)�3�B�D�?�A�9�6��������������@�M�X�W�W�M�J�@�4�'���H�F�;�1�/�+�/�;�G�H�I�I�H�H�H�H�H�H�H�H�ɺź��������������ɺҺֺ������ںֺ�ǈǈ�}�{�y�{ǁǈǔǗǡǫǭǡǡǔǈǈǈǈ�Ľ������������������Ľнսֽ���ݽн��)�����������6�O�hčĚįİĦč�s�V�B�)������!�-�:�_�l�u����������t�l�:�����m�T�J�F�H�[�a�a�m�p����������������z�m��	�����(�4�6�A�G�A�@�4�(����������������Ŀ̿ǿĿ����������������������ܹϹù��������������ùϹܹ�������������������� ������������������������t�r�l�t�t�t�t�t�t�t�t�t����� ��'�@�[�r�������~�r�Y�@�3�����ܻڻӻлû»ûͻлӻۻܻ�޻ܻܻܻܻܻܽ!�������!�.�:�S�l�}�y�v�`�S�G�.�!���������������Ľнؽݽ����۽нĽ��������	�����)�*�+�)���������E�E�E�E�E�E�E�FFF$F)F$FFFFE�E�E�E湑�������������ùܹ����߹ܹй�������ŹŵŭŪŭŭŰŹ��������������������ŹŹ���������������������������������������n�l�l�m�n�v�zÇÐÓØàæëàÓÇ�z�n�n�U�T�J�H�F�H�U�^�a�n�p�z�z�z�n�a�U�U�U�U���������*�6�=�A�6�2�*������������������
��� ���
�����������������ݻл��ûлػܻ�����������������"�'�:�@�M�Y�f�l�s�r�f�Y�M�@�4�'� ; 2 s W 8 8  2 V j ' 2 _ P D b 5 H m = R L 1 9 x j 2 b ; V * M e 2 J P s @ N F S .  _ ; 1 \ ! Z h / < � & $ M � : a 6 ` 6 C + c � B 2 s z ^ * M 7 g - [ ; F      V  X  �  #  a  �  �  �  O  d    k  _    �  �    �  �  �  �  �  �  ,  �  m  �     s  �  �  �  W  m  8  W  �  i  �  L  �  �  �     �  �  �  �  �    �  {  o  �  �  �  �  3  �  >  z  �  �  9  u  [  �  �          D  �    �  �  ��u�D�����
�t�:�o���ě������`B�C������ͼě���/�,1��9X�+�C���j����h��P���
�������������h���ͼ����}��h�ixռ�`B�+�C��� Žt��e`B�]/�\)�,1��㽏\)��w��w�Y��D���P�`��\)�@��'y�#�m�h�y�#��-���w����u�]/��1�m�h�y�#���-�]/���w��hs�m�h��\)���T��+��%��hs��7L��\)�����^5���B�pB6�BL�B!2B4��B��B�B��BҫB rB�B��A�Q�BFIBC�B��B ɺB��A�DB�B ��B0��B��B�MBB�YA���BqB7B�oB�!B&��B)͐B��B�SB��A�B"wB��B�B�[B��B
�B�Bn;B�QB�B�Bm�B\/B�xB)��B�HB#�qB}�B$۶B'(B%n�BHB&��Bl�B>�B�B�Ba�B&j"BBklBq�B�B�B
��BםB�2B6�B�{B�hB��B�?B��B@BLeB ��B4�FB6B�wB��B�GB ��B.^B��A�TXB=�BD>BD�B G�B�A���B<,B ��B0F`Bb�BI�B@B�A���B�UB ,B� B��B&�B)��B@\B�B�qA���B!ͪBASB�UB��BA[B�)B�sB��B��B�#B��B@B@�BP�B)~]B5�B#��Bd�B%J2B�_B%=<B~LB&��BU�B?�B�B&B�B&ŧB.�BAKB��B�aB��B
�DBBD�B8�B�DB�UB�B��A�]�A0��A��GA��AO��B;KA��IA;�IA?�A�reA�
�A�ȅA��A���C�8�A�H'AB�:Abx4A���A�˯A�iiA\F:AI��Ax4AL4A�`�A�@tA}T�@�@��A��A���A��$A��A+;A|��A�v@�ٯ?꾛@Yn�A��A^��A��d?�3�A���AS�?A���A���A��|A�gA�.`@��A���@1?�B/A$�TA؛�@�-�A��A6KbAw2>uteBo7A���?��@�� AoA&}�A�-�C���=��A�d�A毅AɍbAƉA���A���@��@��A���A/~�A�ecA�v�AOZ�BH�A�G�A<A?c�A�N�Â�A�[�Aܤ8A�r�C�?�AƂ�AC>"Ab��A��nA��tA�j�A[BAI�Ay�AKA�z�A�OsA}�@�4�@��A� �A��A���A��A+�A}�A��,@�ϸ?�N�@S��A��pA]�A��?�b�A܅�AR��A��qA��FA���A�A�w'@�d�A�})@0��B@
A#��A�P@{?A�{�A7Av�Z>~m"B�cA���?���@��!AeA&��A��C��F>3�CA���A�y�A�o�A�`+A�HvA���@�{A@إ&   "               !                           $                                       	      	      (      "      	      ;                     %                  "                  /   &            #      
                                    
         (                  %                                    /            !                              3      +            !                                                         3   )   +                  %      #                                                      !                                    -                                          1      +                                                                     3   !   +                  %      #                                    O<"'N�e�Oe�N�x�N�_O�~�O!J�O���N/SBO�ڹO���OE՗O�M�N��O&[*NS��O�P2Oc�iO8<\O1�ON�7N�s!O0&�M�A0N�*zNě{Na��N�NK�sN���PK4UNP6�P8ɍN>��O � N��]O|{�N'�O!��O[
FN}
[O6ؽN�M�N�q�N|�Nm0OU��O8LFOQ��O��O7N�9O��N��O�j�PͩO���P��N�reNxeN!�=N��iNc%�O��~M�%�O��/O%��N_B:N�	O:��N��Nbc�O@�N��N��NT�O:��O���  �  �  <      �  �  �  X    �    �  R  3    R  �     W  ?  V  �  �  :  Z  \  $  '  ?  �  =  �    i  Z  z  q  �  �  �    m  �    _      �  ,  �  p  �  �  c  �  g  9  �  ~  K  �  h  w  �  t  �  �  f  �    �  �  e  �  �  �  �  �<D��<o;�`B;�o;D���D����o���
��o��o�o�t��o�#�
��`B�t��t��D���e`B�D���e`B���㼃o��o��C���t���j��9X��9X��9X���ͼě��ě����ͼě����ͼ�/�0 ż�h�o�+�����+�D���C��C���P�\)��P�@���P���'#�
�#�
�,1�@��49X�8Q�H�9��C��T���Y��T���T���]/�aG��aG��aG��ixսq���q���u�u�y�#��o��o��+���������������������������������������������������������������������������������������������������������	�����������
#')''#
�����?BNZglqrsng[NA?9:<=?��������������������x|��������������zrtx������
#+1/%
������#/<HQUUQPH</-#S[amqr|�����|maTPOQS�������� ���������������������������66ABCOT[`\[[OB;66666��������������������<S[t���worj[B6&_admz��������zmfa``_���
�������vz����������������v$*6COQZ\ab_OC6*� ����������������
��������������������������������
 
�������:;>HT[a^VTIHE=;9::::�����������������MO[]c][UOLMMMMMMMMMMU[`hotnh[\[UUUUUUUUU��������������������#0n{�����nI<0����������������������5BIOO:5)"������#$/73/#ABKNOZ[bgjnng[PNFB?A��������������������������������������X[bhptyth[VUXXXXXXXX�����
��������$)/67?A>)	 .5BNONLMB51.........Z[hqt�������thf[RRUZ����������������������������������������GHSU^anpnla\USNHGGGGCO[hlnih_[OICCCCCCCC����
#%**' 
�����/<HSUY[UH?</#����������������������������������������w{~����������|{xuuxw<BJOS[_f[YOB<<<<<<<<�������	

����������������������������#0<AFFFA0#
�����5:7+&%+<>5)���
#0<DQTRQK<0#
	

y}�������������|y7<IUWVWVUOI=<7447777Zacnusna`\ZZZZZZZZZZNOOX[_hhhh[ONNNNNNNN��������������������������������������������������������~�.0<>EIMUVUID<0......����������$)/0,% �� ��������������������#&*/059;/-#";<HUagnsvynaUSIHD?;;stz�����������}utqss��������������������)40)&����ABNR[^_[YPNBB>7<AAAA05BN[][Z[VNB95010000��������������������������������������������
����������
�����������
��#�/�:�F�K�H�<�/�#��
������������������ ��������������������������� �	��"�*�&�"���������U�N�H�@�<�8�<�A�H�U�Z�a�l�k�i�l�e�a�U�U�����������ʾоѾʾ��������������������������������������$�1�0�3�0�,�$��������A�@�5�3�1�5�;�A�N�Z�g�o�s�v�s�h�g�Z�N�A�����#�(�4�A�M�f�s�x�u�f�Z�M�A�4�(��M�I�M�X�Z�f�s�t�t�s�f�Z�M�M�M�M�M�M�M�MìàÓ�z�n�f�j�n�zÇàì÷����������ùììâàÓÐÎÐÓÚàìù������������õì�����������������������������������������O�B�4�1�B�O�[�h�tāčĚĦĥĜČā�t�h�O��t�s�t�~¢¦©ª¦D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D��a�a�U�T�U�[�a�i�n�z�{�z�x�r�n�b�a�a�a�a�M�H�F�L�M�G�I�U�Z�s��������������f�Z�M�)�����������	��.�T�m�����~�r�m�T�;�)�����������������������	��!��	������������������������+�5�B�D�5�2�)������C�6�*������������*�6�A�O�Q�Q�O�C�	������������	��"�*�.�/�/�(�"��	�����������������������������������������������������������Ŀѿֿٿݿ��ݿѿĿ�����������������������������������������������&�)�6�B�C�B�7�6�.�)������(�$�����(�5�A�D�N�O�N�A�<�5�(�(�(�(�ݿܿӿѿϿѿٿݿ������ݿݿݿݿݿݻ��������ûлԻлʻû��������������������x�u�x�{�����������������x�x�x�x�x�x�x�xàßÝàæìù����������ùìàààààà�������t�G�G�a�s��������������������������	��������������������������������������������������������нϽϽнڽݽ������ݽнннннннпĿÿ������Ŀſѿ׿ݿ������������ݿѿ��/�"��	���	��"�/�7�;�H�T�Y�T�T�H�;�/����'�4�@�M�Y�f�s�z�y�q�f�Y�M�@�4�'��Y�S�Y�\�e�p�r�r�t�x�r�e�Y�Y�Y�Y�Y�Y�Y�Y����޺�������!�(�&�#�!��������s�l�t���������������������������������s�������"�.�8�2�.�"�������������������������������������������������@�@�8�@�H�L�W�Y�e�f�l�e�c�Y�L�@�@�@�@�@�h�g�\�[�Z�[�h�tāčĖĔčĉā�t�h�h�h�h�׾Ӿʾɾʾ˾־׾ؾ������޾׾׾׾����������������������������������������������z�m�c�e�j�m�z���������������������������������������������������������������������������#�0�=�H�0�#��
����������
������������)�3�6�6�7�6�-�)������������@�M�X�W�W�M�J�@�4�'���H�F�;�1�/�+�/�;�G�H�I�I�H�H�H�H�H�H�H�H�ɺȺ������������������ɺκֺ����ֺ�ǈǈ�}�{�y�{ǁǈǔǗǡǫǭǡǡǔǈǈǈǈ�Ľ������������������Ľнսֽ���ݽн��)�����������6�O�hčĚĭįĦč�r�U�B�)������!�-�:�F�S�l�w�|�x�w�l�S�:���m�T�J�F�H�[�a�a�m�p����������������z�m��	�����(�4�6�A�G�A�@�4�(����������������Ŀ̿ǿĿ����������������������Ϲƹù������¹ùϹѹڹҹϹϹϹϹϹϹϹ������������ ������������������������t�s�m�t�t�t�t�t�t�t�t�t�t�t����� ��'�@�[�r�������~�r�Y�@�3�����ܻڻӻлû»ûͻлӻۻܻ�޻ܻܻܻܻܻܽ!�������!�.�:�S�l�}�y�v�`�S�G�.�!�����������������Ľнݽ��ݽؽнĽ��������	�����)�*�+�)���������E�E�E�E�E�E�E�FFF$F)F$FFFFE�E�E�E湝�������������ùϹܹ��޹ܹϹιù�����ŹŵŭŪŭŭŰŹ��������������������ŹŹ���������������������������������������n�l�l�m�n�v�zÇÐÓØàæëàÓÇ�z�n�n�U�T�J�H�F�H�U�^�a�n�p�z�z�z�n�a�U�U�U�U���������*�6�=�A�6�2�*������
����������
������
�
�
�
�
�
�
�
����ݻл��ûлػܻ��������������'��#�,�:�@�M�Y�^�f�k�p�s�p�f�Y�M�@�4�' : . s E 8 ;  6 S j * % Z J D ` 3 K a = S 8 1 9 x j 3 L ; V ( J K 4 J P s 0 N 0 K .  _ " 1 \  Z h  < � % $ M � . a 6 ` # C + c � B , s z T * M 7 g - # ; B    �    X  -  #  �  V    b  O    �  6      �  r  �  	  �  �  �  �  �  ,  �  �  �     s  �  �  l  +  m  8  W  �  i  ]  �  �  �  �  �  �  �  �  �  �  7  �  {  =  �  �  �  �  3  �  >  <  �  f  9  u  [  a  �    �      D  �    f  �  H  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  f  G    �  f  �  ]  �  @  �  W  u  �  �  �  �  x  k  [  H  1    �  �  �  �  �  �  �  �  <  ;  7  0  $      �  �  �  �  x  I    �  �  �  t  '  �  �  �            �  �  �  �  �  �  p  L  &    �  �       	       �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  i  O  9  %  
  �  �  f    �  B   �  �  �  �  �  �  �  �  �  �  p  K    �  �  g  !  �  �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  k  C  !    �  �  �  �  6  2  .  +  3  ?  J  P  S  V  X  Z  [  Y  Q  J  @  .    	      �  �  �  �  �  �  S    �  �  C  �  w  �  h  �  0  �  �  �  �  �  �  �  �  �  �  �  q  W  ;  
  �  �  3  �  �  �  �  �  �  �      �  �  �  �  �  �  y  ^  6  �  �  t  %  �  �  �  �  ~  w  n  c  S  5    �  �  v  =    �  �  6  �  ^    F  O  R  K  <  %  
  �  �  �  k  4  �  �  �  .  �    l  3  &       �  �  �  �  k  ,  �  }    �    k  �  '  b  �  �  �    !  &  %  !    
  �  �  �  �  v  D  �  ~  )  �  s  R  R  J  <  '    �  �  �  �  �  �  d  ,  �  �  z  H    k  �  �  �  �  �  �  �  �  �  �  �  c  b  S  (    �  �  w                   �  �  �  �  x  X  :    �  �  �  �  �  W  M  M  P  S  R  K  C  5      �  �  �  {  3  �  �  V    ?  ?  :  3  *      �  �  �  �  �  Y  .  �  �  �  n  .  �  M  Q  T  U  U  U  R  K  B  6  $  
  �  �  �  |  N  �  \   �  �  �  �  �  z  q  g  ]  T  J  ?  3  '      �  �  �  �  �  �  �  �  �  �  t  g  Y  J  ;  +      �  �  �  �  c  =    :  5  1  ,  (  $                 �   �   �   �   �   �   �  Z  Q  E  8  (      �  �  "  8  5  &  
  �  �  :  �  �  ;  O  P  R  T  W  Z  [  W  P  E  8  )      �  �  �  �  B   �      !  "      
  �  �  �  �  z  P    �  �  �  K     �  '             �  �  �  �  �  �  �  �  �  w  U  3    �  ?  A  D  D  B  >  5  ,    	  �  �  �  �  �  z  d  L  4    �  �  �  �  �  �  �  �  �  l  I  &    �  �  �  ^  -  �  �  7  ;  )    �  �  �  �  c  M  7       �  �  �  f     �  �  �  �  �  �  �  �  �  �  �  �  �  y  a  J  1        �   �   �        �  �  �  �  �  a  5  $      �  �  �  �  �  7  �  i  i  h  g  g  e  ^  W  O  H  >  0  #      �  �  �  �  {  Z  R  I  <  ,      �  �  �  �  �  �  ~  _  9    �  �  }  z  j  Y  H  3    	  �  �  �  �  �  �  �  �  }  q  8  �  �  �  �  '  L  d  p  a  A    �  �  e    �    d  �  
  e  O  �  �  �  �  �  �  z  p  g  Z  N  B  ;  5  5  =  E  J  M  O  �  �  �  �  �  �  g  R  ;    �  �  b    �  �  C  �  |  	  �  �  �  �  �  �  �  �  �  u  T  /    �  �  ^    �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  v  �  �  �  �  �  m  g  b  _  [  V  M  B  0      �  �  �  �  �  }  h  �  �  �  �  �  �  ~  u  m  d  a  h  n  u  o  b  T  G  8  )      �    W  �  �  �  �    
  �  �  �  �  u  A    �  q  �  �  _  W  P  H  ?  -    	  �  �  �  �  �  �  s  b  D  "     �        �  �  �  �  �  �  �  �  �  x  o  g  ^  K  6  !    �          �  �  �  �  �  �  �  f  =    �  �  �  t  -  �  �  �  �  n  X  @  *  	  �  �  �  q  j  l  b  @    �  �  ,       �  �  �  �  �  �  �  d  D     �  �  �  �  ,  �  &  5  W  s  �  �  �  �  �  �  z  X  *  �  �  <  �  P  �    �  p  i  a  V  J  :  -  $          �  �  �  �  �  a  5  	  �  �  �  �  �  �    p  ^  L  :  (       �   �   �   �   |   b  �  �  �  �  �  �  �  �  �  �  �  n  I    �  �  U    �  �  c  O  9    �  �  �  �  _  2    �  �  d    �  x  "  �  l  �  �  o  T  ?  -    $  4  d  `  Q  8    �  �  �  S  �  [  f  c  Q  0    �  �  N  �    .    �  �  2  �  �  �  �   �  �    1  8  6  -       �  �  �  z  ?  �  �  9  �  x  6  �  �  �  �  �  �  s  G    �  �  �  �  l  Q  .    �  �  &  @  ~  u  j  \  J  5      �  �  �  �  S    �  �  A  �  �  8  K  A  7  ,  #  !                              9  f  �  �  �  �  	  G    �  �  �  Y    �  �  5  �  �    h  R  <  &      �  �  �  �  �  �  �  �  q  b  Y  j  {  �  h  o  v  ~  �  �  �  �  �  �  �  �  s  b  G  +    �  �  �  �  �  �  �  l  N  &    �  �  �  w  S  0    �  �  �  H  \  t  u  v  x  y  z  |  }  ~  �  }  u  n  g  _  X  Q  J  B  ;  �  �  �  �  _  >    �  �  �  �  b  ,  �  �  B  �  ^  �  �  �  �  �  �  �  �  �  �  x  \  6    �  �  �  w  W  9    �  f  ^  W  P  H  A  9  0  '          �  �  �  �  �  �  �  �  �  �  ~  ^  ;    �  �  �      �  �  �  s  �  )  �            �  �  �  �  U    �  �  v  :  �  �  &  �  C  �  �  �  �  �  �  �  z  k  \  I  6  "    �  �  �  �  �  Z  $  �  ~  w  q  j  b  V  K  ?  3  (      	  �  �  �  �  �  �  e  U  B  *    �  �  �  �  �  b  @    �  �  �  �    ,  E  �  �  �  �  �  �  �  p  U  :    �  �  �  o  1  �  �  S    �  �  �  �  �  �  �  �  q  Z  D  *    �  �  �  M     2  E  "  n  �  �  �  �  �  �  �  �  �  �  l  R  7       �  �  �  �  �  �  r  Z  N  A  +    �  �  �  U    �  p    �  0  I  �  �  �  �  �  �  �  o  M    �  �  P  �  �  9  �  P  �  �