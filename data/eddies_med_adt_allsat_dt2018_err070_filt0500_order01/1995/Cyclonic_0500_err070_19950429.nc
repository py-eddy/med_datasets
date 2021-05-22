CDF       
      obs    R   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�hr� Ĝ     H  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��G   max       Ph�?     H  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <T��     H   <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>nz�G�   max       @F������     �  !�   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v��z�H     �  .T   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @P�           �  ;$   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ȃ        max       @�1          H  ;�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <#�
     H  =   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B4��     H  >X   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4�g     H  ?�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�8   max       C��     H  @�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >� `   max       C��     H  B0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          E     H  Cx   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     H  D�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3     H  F   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��G   max       POxT     H  GP   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���|���   max       ?�F�]c�f     H  H�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���-   max       <D��     H  I�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>nz�G�   max       @F������     �  K(   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v�
=p��     �  W�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  d�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ȃ        max       @�L�         H  el   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�     H  f�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?�C��$�     �  g�                  
             )      D                        A            	                     +   2               4   	      
         	   	   #   
      	                  *               	         
         
            %      	                           )      N�gNC�N(߆N�D�N%0�N�u�M��GO�`O�TPH�N���P3��OX��Nܡ�O��N��hOs��O� O��BP
��Od�CO��nN�9N;qN�͘N�R�ON*�O��OHҟO:��P� Ph�?N��O�,�N�Z�OA6�P%�N� �N�N��	N��QO(i�N��N��O�αN��8NFW�NOU�O+�zO뫵O��DN��NH��O���O��O��fO(��O��NtUwN)ɯO��N��[NQ(@OU�N��N��xN��N�O���N���N���O���N4f�O�H&O	PN|h�OZ).NF7�O��GO���N�'�N[G6<T��<D��<49X;�`B;ě�;ě�;ě�;��
;��
;D��:�o�D���ě��ě���`B��`B�o�o�T���e`B�e`B�u�u�u��o��o��o��o��C���C���t����㼛�㼛�㼣�
���
��9X��9X�ě����ͼ�`B��`B�����������+�+�C��C��C��t��t��t���P�����#�
�#�
�#�
�#�
�0 Ž0 Ž8Q�D���D���L�ͽaG��e`B�ixսm�h�m�h�m�h�y�#�y�#��%��%������P���P����9<HMUZacnnsvnaUQHB<9��������������������#',000#��������������������������������������������������������������������������������s�������������zxqps����
,/1//'#
���������������������������������)4BN[gt�����t[N5))��
#+>JH</#
����������������������������
#$--#!
������:BIOY[e\[ONB<6::::::������������)05<EMLB6)
����������������������������������������*6COTXZZO6*aaq����������zpmha_a]amrsrmha\X[]]]]]]]]��������������������������

���������#'/8<HPIH</#$/=HUWUTPOIH<6/,#nt���������tutrsnijn��������������������)/6BO[hiotwti[O@6);:=HTamw~���maTMHB;;���
#=Ufhn||~nI<#���������������������������������������������������������������������������������$8GOYRLC6������ghtv������{thc\_ggggY[`ehnih[[VVYYYYYYYY����������������������������������������/28<?FHUaca^]ZUH<://������������������������


����������������
!%"
�����������������������������

�������������������������������,0<ISUVWWSKI<:40+(),s����������������thsv{��������������uqyv������������lmz�����ztmmllllllll(,6@O[hrtqqojh[OB6-(#'<VehhlkbTI<0 ������������������������������������������)*)# ����OO[hhplh[TOLOOOOOOOO[[^`hkstxth[[[[[[[[[�����������������������)0,*�������

����������(0<ITY\UTPI<0#{������������znnnz{{����������������������������������������NN[`gntutgd[YNNNNNNN#/<HPnz���znaUH</#!#jn{{����������{ypnjj���������������������)BNgrthZB5���������������������������)/3)&����otw�������������tpno�������������")33*(#45BFHIFB>55344444444ty��������������~wtt�����������)-12/)"���������������������4�2�,�(�����'�(�4�A�H�K�O�M�F�A�6�4���������������ʾξо˾ʾ��������������������������������ż��������������������������������$��������������������h�c�b�h�uƁƎƖƎƁ�z�u�h�h�h�h�h�h�h�h�������������������	�
�	��������������˻�����!�%�"�!����������������������������*�6�:�A�E�C�6�������:�5�(�%�$�(�5�A�N�Z�h�s�v�w�v�p�g�Z�N�:������Ƨƭ�������0�:�$�� ���$�������n�k�a�U�R�N�T�U�a�h�n�t�zÇÉÇÀ�z�n�n���������������˿ݿ��������ԿѿĿ������%�)�6�B�O�b�W�Q�R�O�H�B�:�6�)�ìáàØÙàäìù����������þùììììàÓËÊÒØßàìù��������������ùìà�U�P�U�X�a�d�n�zÅ�{�z�t�n�a�U�U�U�U�U�U������������������������!�&�������]�M�B�N�Z�f�s����������������������s�]�M�D�M�T�K�L�Z�f�s����������������s�Z�M�Y�'������'�S�Y�f�{�����żռʼ���Y�.�(��	���������	��"�.�9�=�7�;�=�;�.����ùéàÇ�Èàìù�������&�������[�O�Q�[�h�t�|āĆā�t�h�[�[�[�[�[�[�[�[��ݻܻлܻ������������������鿒�����|��������������������������������¿³·¿����������������������������¿¿������������������������� ��	��������׾�ھԾѾ׾����	���"�.�1�.�"�������U�a�n�t�{À�n�e�a�U�S�H�E�=�<�5�4�<�H�U�m�h�`�[�W�N�Q�_�`�m�w�y�z���������}�r�mƎƁ�u�e�^�`�f�uƎƳ��������������ƧƎ���������~�s�c�Z�N�K�g����������� ����������������������$�'����m�`�V�M�H�N�T�`�m�y�����������������y�m�"���	����	��"�/�;�;�H�P�H�<�;�/�'�"�����������������)�>�B�F�N�B�6�)���Y�L�O�X�g���������������������������g�Y�����������������������������������������L�E�L�Y�[�e�r�r�r�r�e�Y�L�L�L�L�L�L�L�L¦¡¦¦²µ¿»²¦¦¦¦ǈ�}�|�{�x�p�y�{ǈǈǔǜǜǖǔǒǈǈǈǈ��������������������������������� �����������z�z�t�v�y�z�{�����������������������z�x�z�}�|�����������������������z�z�z�z���������������������ɺֺ����������ɺ�ŔőňŐŔşŠŭűŸŶŰŭŠŔŔŔŔŔŔ�ܻлл˻лܻ���������ܻܻܻܻܻܻܻ�ìæàÓÒÓàìù��������ùìììììì�����������(�4�A�M�O�P�M�A�6�(���ŇŇŁŁŔťŧŭ������������������ŭŔŇ�T�H�@�?�B�U�a�d�m�z�|����������z�m�a�T�m�j�j�l�m�s�z���~�z�m�m�m�m�m�m�m�m�m�m���
���"�/�/�3�/�$�"����������ܹѹù������ùϹܹ��������������!�����!�-�:�F�S�l�n�_�V�R�I�:�-�$�!�����������5�B�N�T�a�[�N�B�5�)�������������������
���#�.�,��
�����������������������������������������������˻����������ûлӻۻл˻û��������������������������������������������������������ʼ��������z������Լڼ�������ּʼ��������	��!�.�.�-�0�.�!��������Ľý��������ĽнԽݽ߽ݽ۽нĽĽĽĽĽĽ����������|�w�{�����������ĽͽĽ������������������������ĿƿĿ������������������t�s�h�f�c�h�tāčĚĦĪĲĦĚĕčā�t�t�_�^�S�K�F�E�F�O�S�Z�_�l�z�������}�x�l�_����������������������������������������F
F	FFFF<FBFJFcF|F�FyFzFuFlF]FEF$FF
�����	���'�,�4�;�@�J�@�5�4�'�"������������������!�����������������Ç�{�d�`�J�F�C�H�U�a�zÔÚäìøýìÓÇ����������������������������������������Ľ����������������������Ƚн߽��ݽн�����Ŀļ����������������� �� ���������̽.�$�!�����!�.�:�<�F�B�:�.�.�.�.�.�.�5�)������������������)�6�T�[�`�O�5��	�	���*�6�8�6�+�*���������±¦¢²·¿������������������¿±�W�M�B�;�7�@�M�Y�f�r��������������r�f�W�������������"�'�4�4�4�'�'�����/�+�)�/�<�H�P�O�H�<�/�/�/�/�/�/�/�/�/�/ < 6 @ ? j e M ?  O X ; o + - [ 0 ? B d L � X D > i A T : H W f N C m M L B Z J U E k 4 E ) E s = B h [ = 0 e E Y 3 - B ? ~ * 5 p J I � c F A w K 2 p V z [ . + = 0    B  U  �  s  /   �  �      �  C  �  �    �  �  �  �      �  �  K  �  ?  �  �  �  �  �  �      "  �  �  �  I  �    r  �  �  �  �  a  �  t  C  ?  r  \    �  D  �  P  z  @  �  �  U  �  &  &    �  @  �  �  �  U    �  �  e  n  �  Q    ^:�o<#�
;�o%   ;o�D��;�o�T��������㻣�
��hs��1�����󶼣�
�+�+��㽟�w���,1���
�ě���1��`B��h�+�D����`B�u��O߼����+�����8Q콕������\)�C���w��w��㽅��'�w�'e`B�e`B�aG����#�
���-�ixս�7L�]/�<j�D���8Q콏\)�L�ͽL�ͽ�C��]/��O߽y�#�T����j��o������
�}󶽙��������������7L����������Q�B��B4��B%,�BlCB�NB��B+��B ��B��B��BhB	*�B��B|�Br�B�AB�B �B!�B!��B0@�B ��A��B�B�_B �B��B�jB!-B�6A�J&B%�AB*B+4�A���B�OB�IB3oBl�B��B'?Bm�B��BH�B#r+B�xB#�;BH�B&oB�BVB�A���B�,B&YuB�fBCB�xBT7B�`B+$�B.��B#�5B&A~B�Bu�B��B��B�SB)�B��B_ZB��B�B
}�BQ[B��B�B�B5B��B�B1B4�gB%@SB?�B��B��B+�5B �iB��B�xBT�B	@zBo[B��B@�B��B��B��B!C�B!��B0A�A�|�A���B�%B��BGBzB�_B!0�B�~A��OB&>;B*)1B+04A��B��BH�B@�BN�B=(B��BB�B��B1zB#B�{B#�>B6iB&�BƂB>�B,�A�ydB �B%��B��B?`B9�B��BȉB*}�B.̟B#��B&�B��BF�B@mB��B �B)A?B�<B�|B�B?`B
AB<�B�mB�B
��B@DB��B
�iA9��AO�)@�TA1�yB�;A��@d��A�A���Ba�A��A|]�A׃2A��GA��}A���A���AE��AB��@�A]�A���A�ř@�=�AsA���A�"AW�"A���Ak^�B^A��kA�όAlR�A�ʖAՊ�A���A���?�	�A�nKB�_A�1A�}LA�g�@3��A���@�A���A6�A���A���A��A�H>�8@r̬A���A焋A��&@��$@���@��jAz0A('�A!XZAv�A݋�@��A�g�C��@��B��A��tA�pFA$�UA偆A��A�s4A���A��@�)i@�.AÚ�A9�AO�@�SEA2��B3dA�Oj@d6�A���A��/B=�AƣA}uA�gLA�`0A̕�A�q_A�m�AE:!AB��@�S�A[:�Aψ�A�@��As��A��5A��AW �A�~�Ak&dB��A���A��~Ak��A���AՒ�A�uwA��u?�,A�^B�A���A�{�A�m�@3��A��2@��=A�|PA77A�|�A��xA��A�>� `@k��A��A�}�A�7@��@�ޛA �A
�A(�nA �AAv��A�w�@�B�A��@C��@�6)B��Aǀ)A��BA&�yA�gA�A�\A��}A�r@�.�@�oA��0                  
             *      E                        A            
                      +   3               4   
               
   	   $      	   	                  *            	   	                  
            &   	   	                           *                                    /      -                     #   /      )                           '   ;               +                        %               %   !            !                  '                        %         '                                                                                          '      )                           '   3               #                                          !                              %                        %         '                              Nl<�NC�N(߆N�D�N%0�N��M��GO��rO�IO��N�J>Om�OX��N�J�O��N>9uOdlUO� O��jOȟ.O:�RO��nN�9N;qN�͘N�R�ON*�N�6�O �O ��P��POxTN��O�N�Z�O5\O���N� �N�N��	N��QN��N��Nw��O+[)N��8NFW�NOU�O��O~��O��DN��NH��OJ�ZOB�O\��O(��O��NtUwN)ɯO�H�N��[NQ(@O9�CN��N�7�N��N�O�*�N���N���O���N4f�O�H&Nyt�N|h�OZ).NF7�O��GO���N���N[G6  '   �  �  J  �  �  �  �  =  u  R    1      ]  &  j  �  S  �  *    �  B  �  �  �  b  4    8    �  �    |  �  �  )  X  "  |    �  j  �  �  �    �  �  C  y  �  �  �  [  �  �  �  �  �  ;  �    �  X  -  i  =  b     �  y  )  (  4  �  �    <t�<D��<49X;�`B;ě�;��
;ě�;�o���
�#�
%   �\)�ě��o��`B�#�
�t��o��C����ͼ�o�u�u�u��o��o��o��t���1��t����㼴9X���㼼j���
�ě��\)��9X�ě����ͼ�`B�����o�8Q�����+��P�'C��C��t��49X�0 Ž,1�����#�
�#�
�'#�
�0 Ž8Q�8Q�T���D���L�ͽe`B�e`B�ixսm�h�m�h�m�h����y�#��%��%������P���-����HHUXagnnnlaUIHHHHHHH��������������������#',000#��������������������������������������������������������������������������������u�������������|zvsru���
##'&%#
��������������� �����������������������IN[gt����tg[RNLGEDI��
#+>JH</#
����������������������������
#$--#!
������ABDOV[a[OB>9AAAAAAAA������������)05<EMLB6)
������������������������������������������!**6CKOQTVTOG6*aaq����������zpmha_a]amrsrmha\X[]]]]]]]]��������������������������

���������#'/8<HPIH</#$/=HUWUTPOIH<6/,#pt�����������tsmklpp��������������������%)26BO[fhmrhf[OCB6)%:=HTamv~���maTNHB<<:���
#<Unv{{pI<#�����������������������������������������������������������������������������������/8>BDA;6�����ghtv������{thc\_ggggY[`ehnih[[VVYYYYYYYY����������������������������������������<<CHLU[__\[WUH=<45<<�����������������������

������������������

�����������������������������

�������������������������������-07<HIPSUVUQI<610.+-��������������������v{��������������uqyv������������lmz�����ztmmllllllll36;EO[hnpmmjh[OB63.3"#0<IPU^aWUI<90*$#"������������������������������������������)*)# ����OO[hhplh[TOLOOOOOOOO[[^`hkstxth[[[[[[[[[�����������������������)0,*�������

����������#*0;>IRX[TRNI@0&#{������������znnnz{{����������������������������������������NN[`gntutgd[YNNNNNNN#/<HP[nz��}naUH</$!#jn{{����������{ypnjj���������������������)BNgrthZB5���������������������������)/3)&����st����������utssssss�������������")33*(#45BFHIFB>55344444444ty��������������~wtt�����������)+//,)���������������������4�4�*�(�"�(�3�4�9�A�G�I�A�5�4�4�4�4�4�4���������������ʾξо˾ʾ��������������������������������ż��������������������������������$��������������������h�c�b�h�uƁƎƖƎƁ�z�u�h�h�h�h�h�h�h�h�������������������������������������˻�����!�%�"�!����������������������������*�6�9�?�B�6��������A�6�5�/�0�5�A�B�N�Z�g�i�o�n�h�g�Z�N�A�A����ƽƶƼ��������������������������z�q�n�a�U�S�O�U�Z�a�d�n�q�z�}�}�z�z�z�z���������ÿѿݿ�������������ݿѿĿ������%�)�6�B�O�b�W�Q�R�O�H�B�:�6�)�ìæàÚÛàèìùÿ��������úùììììàÓËÊÒØßàìù��������������ùìà�U�T�U�[�a�h�n�u�x�q�n�a�U�U�U�U�U�U�U�U���������������������� �%��������޾]�M�B�N�Z�f�s����������������������s�]�Z�Q�T�X�O�T�Z�s�����������������s�f�Z����Y�@�'��!�'�4�@�Y�f�r������ü�������	������������	��"�(�.�6�9�4�.�"�����ùéàÇ�Èàìù�������&�������[�O�Q�[�h�t�|āĆā�t�h�[�[�[�[�[�[�[�[��ݻܻлܻ������������������鿒�����|��������������������������������¿³·¿����������������������������¿¿������������������������� ��	��������׾�ݾ׾׾Ծ׾�����	�����	�������U�H�G�?�<�8�<�H�U�Z�a�h�n�r�w�{�n�l�a�U�m�k�`�]�Y�P�T�`�c�m�y�������������z�o�mƁ�u�f�_�a�g�uƎƳ���������������ƧƎƁ�������������w�i�h�s������������������������������������$�'����y�s�`�[�T�Q�O�T�]�`�m�y���������������y�"���	����	��"�/�;�;�H�P�H�<�;�/�'�"����������)�5�6�@�B�6�)�������s�g�\�]�i�����������������������������s�����������������������������������������L�E�L�Y�[�e�r�r�r�r�e�Y�L�L�L�L�L�L�L�L¦¡¦¦²µ¿»²¦¦¦¦ǈ�}�|�{�x�p�y�{ǈǈǔǜǜǖǔǒǈǈǈǈ���������������������������������������������z�z�t�v�y�z�{���������������������������������������������������������������ɺɺ����������������ɺκֺܺ����ֺ�ŔőňŐŔşŠŭűŸŶŰŭŠŔŔŔŔŔŔ�ܻлл˻лܻ���������ܻܻܻܻܻܻܻ�ìæàÓÒÓàìù��������ùìììììì��	�� �����%�(�4�A�L�L�A�4�/�(��ŔŒŏŔřŤŭŹ����������������ŹŭŠŔ�T�H�@�?�B�U�a�d�m�z�|����������z�m�a�T�m�j�j�l�m�s�z���~�z�m�m�m�m�m�m�m�m�m�m���
���"�/�/�3�/�$�"���������ܹٹϹù������ùϹܹ���������������ܻ�����!�&�-�:�F�H�K�I�F�@�:�-�+�!������	����)�5�B�I�T�T�N�E�5�)�������������������
���#�.�,��
�����������������������������������������������˻����������ûлӻۻл˻û��������������������������������������������������������ʼ���������������ּ߼߼������ּʼ��������	��!�.�.�-�0�.�!��������Ľý��������ĽнԽݽ߽ݽ۽нĽĽĽĽĽĽ������}�y�y�}�����������ĽʽĽ��������������������������ĿƿĿ�����������������ā�{�t�i�h�f�h�tāčĚĞĚĘďčāāāā�_�^�S�K�F�E�F�O�S�Z�_�l�z�������}�x�l�_����������������������������������������F
F
FFFF$F=FDFJFcF|FxFyFtFkF\FDF$FF
�����	���'�,�4�;�@�J�@�5�4�'�"������������������!�����������������Ç�{�d�`�J�F�C�H�U�a�zÔÚäìøýìÓÇ����������������������������������������Ľ����������������������Ƚн߽��ݽн����������������������������������������ؽ.�$�!�����!�.�:�<�F�B�:�.�.�.�.�.�.�5�)������������������)�6�T�[�`�O�5��	�	���*�6�8�6�+�*���������±¦¢²·¿������������������¿±�W�M�B�;�7�@�M�Y�f�r��������������r�f�W��������������'�(�'��������/�+�)�/�<�H�P�O�H�<�/�/�/�/�/�/�/�/�/�/ < 6 @ ? j ^ M A  % _ 4 o # - X 0 ? 3 g ; � X D > i A 0 . F W a N ? m : H B Z J U F k ' P ) E s 8 ; h [ = 0 T 6 Y 3 - B B ~ *  p 4 I � b F A w K 2 ^ V z [ . + , 0  �  B  U  �  s  �   �  O  K  4  �  �  �  �    p  �  �    S  �  �  �  K  �  ?  �    `  �  �  -    T  "  4  �  �  I  �      �  |  �  �  a  �      ?  r  \  �  U  �  �  P  z  @  �  �  U  �  &  �    �    �  �  �  U    �  �  e  n  �  Q  �  ^  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  	      %  #        �  �  �  �  �  w  7  �  �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  �  �  z  s  g  [  M  =  ,      �  �  �  �  �  j  0   �   �  J  @  6  +       	  �  �  �  �  �  �  �  �  n  O    �  �  �  �  �  �  �  z  U  0    �  �  �  �  �  �  �  �  �  �  �  `  o  ~    }  v  m  `  R  @  +    �  �  �  �  a  ,  �  �  �  �  �  �  �  �  �    z  v  n  b  V  J  >  2  &        �  �  �  x  d  K  -    �  �  �  z  U  /  
  �  �  �  b  /  �  �    "  1  9  =  ;  4  !     �  �  _    �  K  �  F  �  (  �  �    H  l  u  p  c  N  -  �  �  q    �    }  �  p  5  >  G  P  Y  c  l  k  f  a  X  L  @  1      �  �  a    �  1  p  �  �  �  �  �        �  �  z  (  �  P  �  <  Y  1  !    �  �  �  �  �  �  s  T  ,  �  �  |  6  �  �  �  �  �  	          �  �  �  �  y  S  )  �  �  �  ^  (  �  �    �  �  �  �  �  �  �  �  u  ]  C  %  �  �    :  �  �  �  �  +  F  P  Y  [  X  U  P  K  B  3  !    �  �  �  �  O    #  $  #  !      �  �  �  �  �  z  P     �  �  b  �  �   �  j  c  Z  L  ;  ,         �  �  �  b  7    �  �    �    v  s  i  �  z  i  U  >    �  �  ^    �  W  �  �  �  �  �  �  �  ;  R  I  /    �  �  8    �  ~  +  �     K  ~  �  [  �  �  �  �  �  �  �  �  �  �  �  �  j  M  %  �  �  �  �  q  *  !    �  �  �  C  �  �  a    �  I  �  �  6  �  S  D  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  f  Q  ;  "  	  �  �  �  �  �  �  �  �  �  B  6  *        �  �  �  �  �  �  �  �  k  U  A  .      �  �  �  �  �  �  �  �  p  N  &  �  �  �  �  u  0  �  �  O  �  �  �  y  }  r  _  M  :  &    �  �  �  �  �  �  h  C    &  �  �  �  �  |  l  \  I  5    �  �  �  s  2  �  �  Z  
  �  $  L  ^  U  E  )    �  �  �  R    �  �  g  �  t  �  �  %  -  3  /  '      �  �  �  �  �  �  �  �  x  ^  C      �  �  �  �  �  �  �  s  N    �  �  g    �  E  �  J  �  �      7  )    �  �  h  6  �  �  �  �  �  �  {  :  �  �    Z      �  �  �  �  �  �  �  �  �  �  �  �  �  v  g  W  E  3  �  �  �  �  �  �  �  �  �  �  �  �  k  I  %  �  �  �  �  {  �  �  �  �  �  �    h  P  8  !  
  �  �  �  �  �  �  �  �  �  �          
  �  �  �  �    J    �  �  e  &  �  �  �    7  [  u  {  i  M  %  �  �  ^  �  �    �  �  K  y  �  �  �  �  �  �  �  �  �  z  e  K  ,    �  �  �  Q  $      �  �  �  �  �  y  k  ]  O  A  3  %            #  /  ;  G  )    	  �  �  �  �  �  �  g  J  ,    �  �  �  z  ?  �  W  X  K  =  0          �  �  �  �  �  �  �  �  �  }  I     �  �         !             �  �  �  �  �  �  `    �  �  |  |  |  x  t  l  d  U  C  (    �  �  �  �  X    �  }  1                           �  �  �  �  �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a    �  9  �  8  j  \  N  ;  )    �  �  �  �  �  m  M  -  	  �  �  �  '  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  g  P  -  
  �  �  �  g  <  o  �  �  �  �  u  [  >    �  �  �  �  Y  '  �  �  v  8  �  ]  h  j  f  q  |    }  u  b  D    �  �  �  t  6  �  �  W  �  �  �  �  �  �  t  `  F  )  
  �  �  �  �  �  o  c  q    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  _  C  5  &    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  /  i  v  y  v  j  V  9    �  �  �  (  �  ;  �  G  �  �  d  i  t  t  ~  �  �  �  �  �  ~  Y  *  �  �  t  ,  �  �  {  �  �  �  �  �  �  �  �  �  ]  ,  �  �  }  8  �  �  �  �  r  U  �  �  �  �  r  c  Q  9  $    �  �  �  S  2    �    �  0  [  X  U  O  H  @  4  '      �  �  �  �  �  �  �  s  E    �  �  �  �  �  �  �  �  c  D  "  �  �  �  �  z  ^  @    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  A    �  �  K  �  �  W    �  `  C  O  U  ?  (    �  �  �  �  �  �  �  �  x  e    �  �  �  y  m  \  J  5      �  �  �  �  �  v  ]  D  *     �  �  9  9  3  ,  "      �  �  �  �  P    �  �  '  �  +  �  �  j  O  4    �  �  �  �  �  �  �  �  �  ~  _  5  
  �  �  �  �  �  �              �  �  �  i  :    �  �  W  �  �  �  �  b  ;    �  �  �  �  e  4    �  �  \  !  �  �  Y  X  T  Q  N  J  G  D  @  =  9  4  ,  $          �  �  �  )  ,    �  �  �  �  Y    �  �  <  �  �  6  �  ,  �  @  9  i  _  U  P  O  H  3      �  �  �  �  q  O  0     �   �   �  =  6  .  '          
         �  �  �  �  �  �  �  �  b  W  D  *    �  �  �  s  R  -  �  �  �  g  Q  4  �  i  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  U  A  7  3  '    �  �  �  �  �  s  P    {  0  Q  U  Y  `  m  y  y  s  j  \  L  8  $        �  �  �  �  )           �  �  �  �  �  �  �  �  c  B  +      �  �  (    �  �  �  �  �  �  �  �  �  j  O  +    �  �  +  "    4  (        �  �  �  �  �  �    f  M  4    �  �  �  �  �  �  �  �  �  �  z  J    �  o    �  �  �  [     
  K  ^  �  �  �  �  �  f  A    �  �  C  �  z  9  �  �  u  C    �  �         �  �  �  �  �  �  g  A    �  �  L  �  �  U  �    
      �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  $